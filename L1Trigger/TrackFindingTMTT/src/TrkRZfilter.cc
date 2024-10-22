#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/TrackFindingTMTT/interface/TrkRZfilter.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"

#include <array>

using namespace std;

namespace tmtt {

  //=== Initialize configuration parameters, and note eta range covered by sector and phi coordinate of its centre.

  TrkRZfilter::TrkRZfilter(const Settings* settings,
                           unsigned int iPhiSec,
                           unsigned int iEtaReg,
                           float etaMinSector,
                           float etaMaxSector,
                           float phiCentreSector)
      :  // Configuration parameters.
        settings_(settings),
        // Sector number.
        iPhiSec_(iPhiSec),
        iEtaReg_(iEtaReg) {
    // Eta range of sector & phi coord of its centre.
    etaMinSector_ = etaMinSector;
    etaMaxSector_ = etaMaxSector;
    phiCentreSector_ = phiCentreSector;

    // Note that no estimate of the r-z helix params yet exists.
    rzHelix_set_ = false;  // No valid estimate yet.

    // Calculate z coordinate a track would have at given radius if on eta sector boundary.
    chosenRofZ_ = settings->chosenRofZ();
    zTrkMinSector_ = chosenRofZ_ / tan(2. * atan(exp(-etaMinSector_)));
    zTrkMaxSector_ = chosenRofZ_ / tan(2. * atan(exp(-etaMaxSector_)));
    unsigned int zbits = settings->zBits();
    unsigned int rtbits = settings_->rtBits();
    float zrange = settings_->zRange();
    float rtRange = settings_->rtRange();
    float zMultiplier = pow(2., zbits) / zrange;
    float rMultiplier = pow(2., rtbits) / rtRange;

    if (settings_->enableDigitize()) {
      zTrkMinSector_ = floor(zTrkMinSector_ * zMultiplier) / zMultiplier;
      zTrkMaxSector_ = floor(zTrkMaxSector_ * zMultiplier) / zMultiplier;
      chosenRofZ_ = floor(chosenRofZ_ * rMultiplier) / rMultiplier;
    }

    // Assumed length of beam-spot in z.
    beamWindowZ_ = settings->beamWindowZ();

    // Name of r-z track filter algorithm to run.
    rzFilterName_ = settings->rzFilterName();

    // --- Options for Seed filter.
    // Keep stubs compatible with all possible good seed.
    keepAllSeed_ = settings->keepAllSeed();
    //Added resolution for a tracklet-like filter algorithm, beyond that estimated from hit resolution.
    seedResCut_ = settings->seedResCut();
    // Maximum number of seed combinations to bother checking per track candidate.
    maxSeedCombinations_ = settings->maxSeedCombinations();
    // Maximum number of seed combinations consistent with sector (z0,eta) constraints to bother checking per track candidate.
    maxGoodSeedCombinations_ = settings->maxGoodSeedCombinations();
    // Maximum number of seeds that a single stub can be included in.
    maxSeedsPerStub_ = settings->maxSeedsPerStub();
    // Reject tracks whose estimated rapidity from seed filter is inconsistent range of with eta sector. (Kills some duplicate tracks).
    zTrkSectorCheck_ = settings->zTrkSectorCheck();

    // For debugging
    minNumMatchLayers_ = settings->minNumMatchLayers();
  }

  //=== Filters track candidates (found by the r-phi Hough transform), removing inconsistent stubs from the tracks,
  //=== also killing some of the tracks altogether if they are left with too few stubs.
  //=== Also adds an estimate of r-z helix parameters to the selected track objects, if the filters used provide this.

  list<L1track3D> TrkRZfilter::filterTracks(const list<L1track2D>& tracks) {
    list<L1track3D> filteredTracks;

    for (const L1track2D& trkIN : tracks) {
      const vector<Stub*>& stubs = trkIN.stubs();  // stubs assigned to track

      // Declare this to be worth keeping (for now).
      bool trackAccepted = true;

      // Note that no estimate of the r-z helix params of this track yet exists.
      rzHelix_set_ = false;  // No valid estimate yet.

      // Digitize stubs for r-z filter if required.
      if (settings_->enableDigitize()) {
        for (Stub* s : stubs) {
          s->digitize(iPhiSec_, Stub::DigiStage::SF);
        }
      }

      //--- Filter stubs assigned to track, checking they are consistent with requested criteria.

      // Get debug printout for specific regions.
      bool print = false;

      vector<Stub*> filteredStubs = stubs;
      if (rzFilterName_ == "SeedFilter") {
        filteredStubs = this->seedFilter(filteredStubs, trkIN.qOverPt(), print);
      } else {
        throw cms::Exception("BadConfig") << "TrkRzFilter: ERROR unknown r-z track filter requested: " << rzFilterName_;
      }

      // Check if track still has stubs in enough layers after filter.

      unsigned int numLayersAfterFilters = Utility::countLayers(settings_, filteredStubs);
      if (numLayersAfterFilters < settings_->minFilterLayers())
        trackAccepted = false;
      //if (numLayersAfterFilters < Utility::numLayerCut(Utility::AlgoStep::SEED, settings_, iPhiSec_, iEtaReg_, std::abs(trkIN.qOverPt()))) trackAccepted = false;

      if (trackAccepted) {
        // Estimate r-z helix parameters from centre of eta-sector if no estimate provided by r-z filter.
        if (!rzHelix_set_)
          this->estRZhelix();

        pair<float, float> helixRZ(rzHelix_z0_, rzHelix_tanL_);

        // Create copy of original track, except now using its filtered stubs, to be added to filteredTrack collection.
        filteredTracks.emplace_back(settings_,
                                    filteredStubs,
                                    trkIN.cellLocationHT(),
                                    trkIN.helix2D(),
                                    helixRZ,
                                    trkIN.iPhiSec(),
                                    trkIN.iEtaReg(),
                                    trkIN.optoLinkID(),
                                    trkIN.mergedHTcell());
      }
    }

    return filteredTracks;
  }

  //=== Use Seed Filter to produce a filtered collection of stubs on this track candidate that are consistent with a straight line
  //=== in r-z using tracklet algo.

  vector<Stub*> TrkRZfilter::seedFilter(const std::vector<Stub*>& stubs, float trkQoverPt, bool print) {
    unsigned int numLayers;                //Num of Layers in the cell after that filter has been applied
    std::vector<Stub*> filtStubs = stubs;  // Copy stubs vector in filtStubs
    bool FirstSeed = true;
    //Allowed layers for the first & second seeding stubs
    constexpr std::array<unsigned int, 8> FirstSeedLayers = {{1, 2, 11, 21, 3, 12, 22, 4}};
    constexpr std::array<unsigned int, 10> SecondSeedLayers = {{1, 2, 11, 3, 21, 22, 12, 23, 13, 4}};
    set<Stub*> uniqueFilteredStubs;

    unsigned int numSeedCombinations = 0;      // Counter for number of seed combinations considered.
    unsigned int numGoodSeedCombinations = 0;  // Counter for seed combinations with z0 within beam spot length.
    vector<Stub*> filteredStubs;               // Filter Stubs vector to be returned

    unsigned int oldNumLay = 0;  //Number of Layers counter, used to keep the seed with more layers

    auto orderByLayer = [](const Stub* a, const Stub* b) { return bool(a->layerId() < b->layerId()); };
    std::sort(filtStubs.begin(), filtStubs.end(), orderByLayer);

    // Loop over stubs in the HT Cell
    for (Stub* s0 : filtStubs) {
      // Select the first available seeding stub (r<70)
      if (s0->psModule() && std::find(std::begin(FirstSeedLayers), std::end(FirstSeedLayers), s0->layerId()) !=
                                std::end(FirstSeedLayers)) {
        unsigned int numSeedsPerStub = 0;

        for (Stub* s1 : filtStubs) {
          if (numGoodSeedCombinations < maxGoodSeedCombinations_ && numSeedCombinations < maxSeedCombinations_ &&
              numSeedsPerStub < maxSeedsPerStub_) {
            // Select the second seeding stub (r<90)
            if (s1->psModule() && s1->layerId() > s0->layerId() &&
                std::find(std::begin(SecondSeedLayers), std::end(SecondSeedLayers), s1->layerId()) !=
                    std::end(SecondSeedLayers)) {
              numSeedsPerStub++;
              numSeedCombinations++;  //Increase filter cycles counter
              if (print)
                PrintL1trk() << "s0: "
                             << "z: " << s0->z() << ", r: " << s0->r() << ", id:" << s0->layerId() << " ****** s1: "
                             << "z: " << s1->z() << ", r: " << s1->r() << ", id:" << s1->layerId();
              //double sumSeedDist = 0.;
              //double oldSumSeedDist = 1000000.;  //Define variable used to estimate the quality of seeds
              vector<Stub*> tempStubs;  //Create a temporary container for stubs
              tempStubs.push_back(s0);  //Store the first seeding stub in the temporary container
              tempStubs.push_back(s1);  //Store the second seeding stub in the temporary container

              // Estimate a value of z at the beam spot using the two seeding stubs
              double z0 = s1->z() + (-s1->z() + s0->z()) * s1->r() / (s1->r() - s0->r());
              // COMMENTED OUT CODE GIVES OPTION OF ALLOWING FOR UNCERTAINTY. NOT WORTH IT?
              //double z0err = s1->sigmaZ() + ( s1->sigmaZ() + s0->sigmaZ() )*s1->r()/std::abs(s1->r()-s0->r()) + std::abs(-s1->z()+s0->z())*(s1->sigmaR()*std::abs(s1->r()-s0->r()) + s1->r()*(s1->sigmaR() + s0->sigmaR()) )/((s1->r()-s0->r())*(s1->r()-s0->r()));
              // Estimate a value of z at a chosen Radius using the two seeding stubs
              float zTrk = s1->z() + (-s1->z() + s0->z()) * (s1->r() - chosenRofZ_) / (s1->r() - s0->r());
              // float zTrkErr = s1->sigmaZ() + ( s1->sigmaZ() + s0->sigmaZ() )*std::abs(s1->r()-chosenRofZ_)/std::abs(s1->r()-s0->r()) + std::abs(-s1->z()+s0->z())*(s1->sigmaR()*std::abs(s1->r()-s0->r()) + std::abs(s1->r()-chosenRofZ_)*(s1->sigmaR() + s0->sigmaR()) )/((s1->r()-s0->r())*(s1->r()-s0->r()));
              float leftZtrk = zTrk * std::abs(s1->r() - s0->r());
              float rightZmin = zTrkMinSector_ * std::abs(s1->r() - s0->r());
              float rightZmax = zTrkMaxSector_ * std::abs(s1->r() - s0->r());

              // If z0 is within the beamspot range loop over the other stubs in the cell
              //if (std::abs(z0)<=beamWindowZ_+z0err) {
              if (std::abs(z0) <= beamWindowZ_) {
                // Check track r-z helix parameters are consistent with it being assigned to current rapidity sector (kills duplicates due to overlapping sectors).
                // if ( (! zTrkSectorCheck_) || (zTrk > zTrkMinSector_ - zTrkErr && zTrk < zTrkMaxSector_ + zTrkErr) ) {
                if ((!zTrkSectorCheck_) || (leftZtrk > rightZmin && leftZtrk < rightZmax)) {
                  numGoodSeedCombinations++;
                  unsigned int LiD = 0;  //Store the layerId of the stub (KEEP JUST ONE STUB PER LAYER)
                  //                                double oldseed = 1000.; //Store the seed value of the current stub (KEEP JUST ONE STUB PER LAYER)

                  // Loop over stubs in vector different from the seeding stubs
                  for (Stub* s : filtStubs) {
                    // if(s!= s0 && s!= s1){
                    // Calculate the seed and its tolerance
                    double seedDist =
                        (s->z() - s1->z()) * (s1->r() - s0->r()) - (s->r() - s1->r()) * (s1->z() - s0->z());
                    double seedDistRes = (s->sigmaZ() + s1->sigmaZ()) * std::abs(s1->r() - s0->r()) +
                                         (s->sigmaR() + s1->sigmaR()) * std::abs(s1->z() - s0->z()) +
                                         (s0->sigmaZ() + s1->sigmaZ()) * std::abs(s->r() - s1->r()) +
                                         (s0->sigmaR() + s1->sigmaR()) * std::abs(s->z() - s1->z());
                    seedDistRes *= seedResCut_;  // Scale cut by this amount.
                    //If seed is lower than the tolerance push back the stub (KEEP JUST ONE STUB PER LAYER, NOT ENABLED BY DEFAULT)
                    if (std::abs(seedDist) <= seedDistRes) {
                      if (s->layerId() != LiD and s->layerId() != s0->layerId() and s->layerId() != s1->layerId()) {
                        tempStubs.push_back(s);
                        LiD = s->layerId();
                      }
                    }
                  }
                }
              }

              numLayers = Utility::countLayers(
                  settings_, tempStubs);  // Count the number of layers in the temporary stubs container

              // Check if the current seed has more layers then the previous one (Keep the best seed)
              if (keepAllSeed_ == false) {
                if (numLayers > oldNumLay) {
                  // Check if the current seed has better quality than the previous one
                  //if(sumSeedDist < oldSumSeedDist){
                  filteredStubs =
                      tempStubs;  //Copy the temporary stubs vector in the filteredStubs vector, which will be returned
                                  //  oldSumSeedDist = sumSeedDist; //Update value of oldSumSeedDist
                  oldNumLay = numLayers;                                      //Update value of oldNumLay
                  rzHelix_z0_ = z0;                                           //Store estimated z0
                  rzHelix_tanL_ = (s1->z() - s0->z()) / (s1->r() - s0->r());  // Store estimated tanLambda
                  rzHelix_set_ = true;
                }
                //}
              } else {
                // Check if the current seed satisfies the minimum layers requirement (Keep all seed algorithm)
                if (numLayers >= Utility::numLayerCut(
                                     Utility::AlgoStep::SEED, settings_, iPhiSec_, iEtaReg_, std::abs(trkQoverPt))) {
                  uniqueFilteredStubs.insert(tempStubs.begin(), tempStubs.end());  //Insert the uniqueStub set

                  // If these are the first seeding stubs store the values of z0 and tanLambda
                  if (FirstSeed) {
                    FirstSeed = false;
                    rzHelix_z0_ = z0;                                           //Store estimated z0
                    rzHelix_tanL_ = (s1->z() - s0->z()) / (s1->r() - s0->r());  // Store estimated tanLambda
                    rzHelix_set_ = true;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Copy stubs from the uniqueFilteredStubs set to the filteredStubs vector (Keep all seed algorithm)
    if (keepAllSeed_ == true) {
      for (Stub* stub : uniqueFilteredStubs) {
        filteredStubs.push_back(stub);
      }
    }

    // Note number of seed combinations used for this track.
    numSeedCombsPerTrk_.push_back(numSeedCombinations);
    numGoodSeedCombsPerTrk_.push_back(numGoodSeedCombinations);

    return filteredStubs;  // Return the filteredStubs vector
  }

  // Estimate r-z helix parameters from centre of eta-sector if no better estimate provided by r-z filter.

  void TrkRZfilter::estRZhelix() {
    rzHelix_z0_ = 0.;
    rzHelix_tanL_ = 0.5 * (1 / tan(2 * atan(exp(-etaMinSector_))) + 1 / tan(2 * atan(exp(-etaMaxSector_))));
    rzHelix_set_ = true;
  }

}  // namespace tmtt
