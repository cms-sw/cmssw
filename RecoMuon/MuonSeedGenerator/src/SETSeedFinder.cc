/** \class SETSeedFinder
    I. Bloch, E. James, S. Stoynev
 */

#include "RecoMuon/MuonSeedGenerator/src/SETSeedFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TMath.h"

using namespace edm;
using namespace std;

const string metname = "Muon|RecoMuon|SETMuonSeedFinder";

SETSeedFinder::SETSeedFinder(const ParameterSet& parameterSet) : MuonSeedVFinder() {
  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("SETTrajBuilderParameters");
  apply_prePruning = trajectoryBuilderParameters.getParameter<bool>("Apply_prePruning");
  // load pT seed parameters
  thePtExtractor = new MuonSeedPtExtractor(trajectoryBuilderParameters);
}

void SETSeedFinder::seeds(const MuonRecHitContainer& cluster, std::vector<TrajectorySeed>& result) {}

// there is an existing sorter somewhere in the CMSSW code (I think) - delete that
namespace {
  struct sorter {
    sorter() {}
    bool operator()(MuonTransientTrackingRecHit::MuonRecHitPointer const& hit_1,
                    MuonTransientTrackingRecHit::MuonRecHitPointer const& hit_2) const {
      return (hit_1->globalPosition().mag2() < hit_2->globalPosition().mag2());
    }
  };  // smaller first
  const sorter sortSegRadius;
}  // namespace

std::vector<SETSeedFinder::MuonRecHitContainer> SETSeedFinder::sortByLayer(MuonRecHitContainer& cluster) const {
  stable_sort(cluster.begin(), cluster.end(), sortSegRadius);
  //---- group hits in detector layers (if in same layer); the idea is that
  //---- some hits could not belong to a track simultaneously - these will be in a
  //---- group; two hits from one and the same group will not go to the same track
  std::vector<MuonRecHitContainer> MuonRecHitContainer_perLayer;
  if (!cluster.empty()) {
    int iHit = 0;
    MuonRecHitContainer hitsInThisLayer;
    hitsInThisLayer.push_back(cluster[iHit]);
    DetId detId = cluster[iHit]->hit()->geographicalId();
    const GeomDet* geomDet = theService->trackingGeometry()->idToDet(detId);
    while (iHit < int(cluster.size()) - 1) {
      DetId detId_2 = cluster[iHit + 1]->hit()->geographicalId();
      const GlobalPoint gp_nextHit = cluster[iHit + 1]->globalPosition();

      // this is the distance of the "second" hit to the "first" detector (containing the "first hit")
      float distanceToDetector = fabs(geomDet->surface().localZ(gp_nextHit));

      //---- hits from DT and CSC  could be very close in angle but incosistent with
      //---- belonging to a common track (and these are different surfaces);
      //---- also DT (and CSC now - 090822) hits from a station (in a pre-cluster) should be always in a group together;
      //---- take this into account and put such hits in a group together

      bool specialCase = false;
      if (detId.subdetId() == MuonSubdetId::DT && detId_2.subdetId() == MuonSubdetId::DT) {
        DTChamberId dtCh(detId);
        DTChamberId dtCh_2(detId_2);
        specialCase = (dtCh.station() == dtCh_2.station());
      } else if (detId.subdetId() == MuonSubdetId::CSC && detId_2.subdetId() == MuonSubdetId::CSC) {
        CSCDetId cscCh(detId);
        CSCDetId cscCh_2(detId_2);
        specialCase = (cscCh.station() == cscCh_2.station() && cscCh.ring() == cscCh_2.ring());
      }

      if (distanceToDetector < 0.001 || true == specialCase) {  // hardcoded value - remove!
        hitsInThisLayer.push_back(cluster[iHit + 1]);
      } else {
        specialCase = false;
        if (((cluster[iHit]->isDT() && cluster[iHit + 1]->isCSC()) ||
             (cluster[iHit]->isCSC() && cluster[iHit + 1]->isDT())) &&
            //---- what is the minimal distance between a DT and a CSC hit belonging
            //---- to a common track? (well, with "reasonable" errors; put 10 cm for now)
            fabs(cluster[iHit + 1]->globalPosition().mag() - cluster[iHit]->globalPosition().mag()) < 10.) {
          hitsInThisLayer.push_back(cluster[iHit + 1]);
          // change to Stoyan - now we also update the detID here... give it a try. IBL 080905
          detId = cluster[iHit + 1]->hit()->geographicalId();
          geomDet = theService->trackingGeometry()->idToDet(detId);
        } else if (!specialCase) {
          //---- put the group of hits in the vector (containing the groups of hits)
          //---- and continue with next layer (group)
          MuonRecHitContainer_perLayer.push_back(hitsInThisLayer);
          hitsInThisLayer.clear();
          hitsInThisLayer.push_back(cluster[iHit + 1]);
          detId = cluster[iHit + 1]->hit()->geographicalId();
          geomDet = theService->trackingGeometry()->idToDet(detId);
        }
      }
      ++iHit;
    }
    MuonRecHitContainer_perLayer.push_back(hitsInThisLayer);
  }
  return MuonRecHitContainer_perLayer;
}
//
void SETSeedFinder::limitCombinatorics(std::vector<MuonRecHitContainer>& MuonRecHitContainer_perLayer) {
  const int maximumNumberOfCombinations = 1000000;
  unsigned nLayers = MuonRecHitContainer_perLayer.size();
  if (1 == nLayers) {
    return;
  }
  // maximal number of (segment) layers would be upto ~12; see next function
  // below is just a quick fix for a rare "overflow"
  if (MuonRecHitContainer_perLayer.size() > 15) {
    MuonRecHitContainer_perLayer.resize(1);
    return;
  }

  std::vector<double> sizeOfLayer(nLayers);
  //std::cout<<" nLayers = "<<nLayers<<std::endl;
  double nAllCombinations = 1.;
  for (unsigned int i = 0; i < nLayers; ++i) {
    //std::cout<<" i = "<<i<<" size = "<<MuonRecHitContainer_perLayer.at(i).size()<<std::endl;
    sizeOfLayer.at(i) = MuonRecHitContainer_perLayer.at(i).size();
    nAllCombinations *= MuonRecHitContainer_perLayer.at(i).size();
  }
  //std::cout<<"nAllCombinations = "<<nAllCombinations<<std::endl;
  //---- Erase most busy detector layers until we get less than maximumNumberOfCombinations combinations
  while (nAllCombinations > float(maximumNumberOfCombinations)) {
    std::vector<double>::iterator maxEl_it = max_element(sizeOfLayer.begin(), sizeOfLayer.end());
    int maxEl = maxEl_it - sizeOfLayer.begin();
    nAllCombinations /= MuonRecHitContainer_perLayer.at(maxEl).size();
    MuonRecHitContainer_perLayer.erase(MuonRecHitContainer_perLayer.begin() + maxEl);
    sizeOfLayer.erase(sizeOfLayer.begin() + maxEl);
  }
  return;
}
//
std::vector<SETSeedFinder::MuonRecHitContainer> SETSeedFinder::findAllValidSets(
    const std::vector<SETSeedFinder::MuonRecHitContainer>& MuonRecHitContainer_perLayer) {
  std::vector<MuonRecHitContainer> allValidSets;
  // build all possible combinations (i.e valid sets; the algorithm name is after this feature -
  // SET algorithm)
  //
  // ugly... use recursive function?!
  // or implement Ingo's suggestion (a la ST)
  unsigned nLayers = MuonRecHitContainer_perLayer.size();
  if (1 == nLayers) {
    return allValidSets;
  }
  MuonRecHitContainer validSet;
  unsigned int iPos0 = 0;
  std::vector<unsigned int> iLayer(12);  // could there be more than 11 layers?
  std::vector<unsigned int> size(12);
  if (iPos0 < nLayers) {
    size.at(iPos0) = MuonRecHitContainer_perLayer.at(iPos0).size();
    for (iLayer[iPos0] = 0; iLayer[iPos0] < size[iPos0]; ++iLayer[iPos0]) {
      validSet.clear();
      validSet.push_back(MuonRecHitContainer_perLayer[iPos0][iLayer[iPos0]]);
      unsigned int iPos1 = 1;
      if (iPos1 < nLayers) {
        size.at(iPos1) = MuonRecHitContainer_perLayer.at(iPos1).size();
        for (iLayer[iPos1] = 0; iLayer[iPos1] < size[iPos1]; ++iLayer[iPos1]) {
          validSet.resize(iPos1);
          validSet.push_back(MuonRecHitContainer_perLayer[iPos1][iLayer[iPos1]]);
          unsigned int iPos2 = 2;
          if (iPos2 < nLayers) {
            size.at(iPos2) = MuonRecHitContainer_perLayer.at(iPos2).size();
            for (iLayer[iPos2] = 0; iLayer[iPos2] < size[iPos2]; ++iLayer[iPos2]) {
              validSet.resize(iPos2);
              validSet.push_back(MuonRecHitContainer_perLayer[iPos2][iLayer[iPos2]]);
              unsigned int iPos3 = 3;
              if (iPos3 < nLayers) {
                size.at(iPos3) = MuonRecHitContainer_perLayer.at(iPos3).size();
                for (iLayer[iPos3] = 0; iLayer[iPos3] < size[iPos3]; ++iLayer[iPos3]) {
                  validSet.resize(iPos3);
                  validSet.push_back(MuonRecHitContainer_perLayer[iPos3][iLayer[iPos3]]);
                  unsigned int iPos4 = 4;
                  if (iPos4 < nLayers) {
                    size.at(iPos4) = MuonRecHitContainer_perLayer.at(iPos4).size();
                    for (iLayer[iPos4] = 0; iLayer[iPos4] < size[iPos4]; ++iLayer[iPos4]) {
                      validSet.resize(iPos4);
                      validSet.push_back(MuonRecHitContainer_perLayer[iPos4][iLayer[iPos4]]);
                      unsigned int iPos5 = 5;
                      if (iPos5 < nLayers) {
                        size.at(iPos5) = MuonRecHitContainer_perLayer.at(iPos5).size();
                        for (iLayer[iPos5] = 0; iLayer[iPos5] < size[iPos5]; ++iLayer[iPos5]) {
                          validSet.resize(iPos5);
                          validSet.push_back(MuonRecHitContainer_perLayer[iPos5][iLayer[iPos5]]);
                          unsigned int iPos6 = 6;
                          if (iPos6 < nLayers) {
                            size.at(iPos6) = MuonRecHitContainer_perLayer.at(iPos6).size();
                            for (iLayer[iPos6] = 0; iLayer[iPos6] < size[iPos6]; ++iLayer[iPos6]) {
                              validSet.resize(iPos6);
                              validSet.push_back(MuonRecHitContainer_perLayer[iPos6][iLayer[iPos6]]);
                              unsigned int iPos7 = 7;
                              if (iPos7 < nLayers) {
                                size.at(iPos7) = MuonRecHitContainer_perLayer.at(iPos7).size();
                                for (iLayer[iPos7] = 0; iLayer[iPos7] < size[iPos7]; ++iLayer[iPos7]) {
                                  validSet.resize(iPos7);
                                  validSet.push_back(MuonRecHitContainer_perLayer[iPos7][iLayer[iPos7]]);
                                  unsigned int iPos8 = 8;
                                  if (iPos8 < nLayers) {
                                    size.at(iPos8) = MuonRecHitContainer_perLayer.at(iPos8).size();
                                    for (iLayer[iPos8] = 0; iLayer[iPos8] < size[iPos8]; ++iLayer[iPos8]) {
                                      validSet.resize(iPos8);
                                      validSet.push_back(MuonRecHitContainer_perLayer[iPos8][iLayer[iPos8]]);
                                      unsigned int iPos9 = 9;
                                      if (iPos9 < nLayers) {
                                        size.at(iPos9) = MuonRecHitContainer_perLayer.at(iPos9).size();
                                        for (iLayer[iPos9] = 0; iLayer[iPos9] < size[iPos9]; ++iLayer[iPos9]) {
                                          validSet.resize(iPos9);
                                          validSet.push_back(MuonRecHitContainer_perLayer[iPos9][iLayer[iPos9]]);
                                          unsigned int iPos10 = 10;
                                          if (iPos10 < nLayers) {
                                            size.at(iPos10) = MuonRecHitContainer_perLayer.at(iPos10).size();
                                            for (iLayer[iPos10] = 0; iLayer[iPos10] < size[iPos10]; ++iLayer[iPos10]) {
                                              validSet.resize(iPos10);
                                              validSet.push_back(MuonRecHitContainer_perLayer[iPos10][iLayer[iPos10]]);
                                              unsigned int iPos11 = 11;  // more?
                                              if (iPos11 < nLayers) {
                                                size.at(iPos11) = MuonRecHitContainer_perLayer.at(iPos11).size();
                                                for (iLayer[iPos11] = 0; iLayer[iPos11] < size[iPos11];
                                                     ++iLayer[iPos11]) {
                                                }
                                              } else {
                                                allValidSets.push_back(validSet);
                                              }
                                            }
                                          } else {
                                            allValidSets.push_back(validSet);
                                          }
                                        }
                                      } else {
                                        allValidSets.push_back(validSet);
                                      }
                                    }
                                  } else {
                                    allValidSets.push_back(validSet);
                                  }
                                }
                              } else {
                                allValidSets.push_back(validSet);
                              }
                            }
                          } else {
                            allValidSets.push_back(validSet);
                          }
                        }
                      } else {
                        allValidSets.push_back(validSet);
                      }
                    }
                  } else {
                    allValidSets.push_back(validSet);
                  }
                }
              } else {
                allValidSets.push_back(validSet);
              }
            }
          } else {
            allValidSets.push_back(validSet);
          }
        }
      } else {
        allValidSets.push_back(validSet);
      }
    }
  } else {
    allValidSets.push_back(validSet);
  }
  return allValidSets;
}

std::pair<int, int>  // or <bool, bool>
SETSeedFinder::checkAngleDeviation(double dPhi_1, double dPhi_2) const {
  // Two conditions:
  // a) deviations should be only to one side (above some absolute value cut to avoid
  //    material effects; this should be refined)
  // b) deviatiation in preceding steps should be bigger due to higher magnetic field
  //    (again - a minimal value cut should be in place; this also should account for
  //     the small (Z) distances in overlaping CSC chambers)

  double mult = dPhi_1 * dPhi_2;
  int signVal = 1;
  if (fabs(dPhi_1) < fabs(dPhi_2)) {
    signVal = -1;
  }
  int signMult = -1;
  if (mult > 0)
    signMult = 1;
  std::pair<int, int> sign;
  sign = make_pair(signVal, signMult);

  return sign;
}

void SETSeedFinder::validSetsPrePruning(std::vector<SETSeedFinder::MuonRecHitContainer>& allValidSets) {
  //---- this actually is a pre-pruning; it does not include any fit information;
  //---- it is intended to remove only very "wild" segments from a set;
  //---- no "good" segment is to be lost (otherwise - widen the parameters)

  for (unsigned int iSet = 0; iSet < allValidSets.size(); ++iSet) {
    pre_prune(allValidSets[iSet]);
  }
}

void SETSeedFinder::pre_prune(SETSeedFinder::MuonRecHitContainer& validSet) const {
  unsigned nHits = validSet.size();
  if (nHits > 3) {  // to decide we need at least 4 measurements
    // any information could be used to make a decision for pruning
    // maybe dPhi (delta Phi) is enough
    std::vector<double> dPhi;
    double dPhi_tmp;
    bool wildCandidate;
    int pruneHit_tmp;

    for (unsigned int iHit = 1; iHit < nHits; ++iHit) {
      dPhi_tmp = validSet[iHit]->globalPosition().phi() - validSet[iHit - 1]->globalPosition().phi();
      dPhi.push_back(dPhi_tmp);
    }
    std::vector<int> pruneHit;
    //---- loop over all the hits in a set

    for (unsigned int iHit = 0; iHit < nHits; ++iHit) {
      double dPHI_MIN = 0.02;  //?? hardcoded - remove it
      if (iHit) {
        // if we have to remove the very first hit (iHit == 0) then
        // we'll probably be in trouble already
        wildCandidate = false;
        // actually 2D is bad only if not r-phi... Should I refine it?
        // a hit is a candidate for pruning only if dPhi > dPHI_MIN;
        // pruning decision is based on combination of hits characteristics
        if (4 == validSet[iHit - 1]->dimension() && 4 == validSet[iHit]->dimension() &&
            fabs(validSet[iHit]->globalPosition().phi() - validSet[iHit - 1]->globalPosition().phi()) > dPHI_MIN) {
          wildCandidate = true;
        }
        pruneHit_tmp = -1;
        if (wildCandidate) {
          // OK - this couple doesn't look good (and is from 4D segments); proceed...
          if (1 == iHit) {  // the first  and the last hits are special case
            if (4 == validSet[iHit + 1]->dimension() && 4 == validSet[iHit + 2]->dimension()) {  //4D?
              // is the picture better if we remove the second hit?
              dPhi_tmp = validSet[iHit + 1]->globalPosition().phi() - validSet[iHit - 1]->globalPosition().phi();
              // is the deviation what we expect (sign, not magnitude)?
              std::pair<int, int> sign = checkAngleDeviation(dPhi_tmp, dPhi[2]);
              if (1 == sign.first && 1 == sign.second) {
                pruneHit_tmp = iHit;  // mark the hit 1 for removing
              }
            }
          } else if (iHit > 1 && iHit < validSet.size() - 1) {
            if (4 == validSet[0]->dimension() &&  // we rely on the first (most important) couple
                4 == validSet[1]->dimension() && pruneHit.back() != int(iHit - 1) &&
                pruneHit.back() != 1) {  // check if hits are already marked
              // decide which of the two hits should be removed (if any; preferably the outer one i.e.
              // iHit rather than iHit-1); here - check what we get by removing iHit
              dPhi_tmp = validSet[iHit + 1]->globalPosition().phi() - validSet[iHit - 1]->globalPosition().phi();
              // first couple is most important anyway so again compare to it
              std::pair<int, int> sign = checkAngleDeviation(dPhi[0], dPhi_tmp);
              if (1 == sign.first && 1 == sign.second) {
                pruneHit_tmp = iHit;  // mark the hit iHit for removing
              } else {                // iHit is not to be removed; proceed...
                // what if we remove (iHit - 1) instead of iHit?
                dPhi_tmp = validSet[iHit + 1]->globalPosition().phi() - validSet[iHit]->globalPosition().phi();
                std::pair<int, int> sign = checkAngleDeviation(dPhi[0], dPhi_tmp);
                if (1 == sign.first && 1 == sign.second) {
                  pruneHit_tmp = iHit - 1;  // mark the hit (iHit -1) for removing
                }
              }
            }
          } else {
            // the last hit: if picture is not good - remove it
            if (pruneHit.size() > 1 && pruneHit[pruneHit.size() - 1] < 0 && pruneHit[pruneHit.size() - 2] < 0) {
              std::pair<int, int> sign = checkAngleDeviation(dPhi[dPhi.size() - 2], dPhi[dPhi.size() - 1]);
              if (-1 == sign.first && -1 == sign.second) {  // here logic is a bit twisted
                pruneHit_tmp = iHit;                        // mark the last hit for removing
              }
            }
          }
        }
        pruneHit.push_back(pruneHit_tmp);
      }
    }
    // }
    // actual pruning
    for (unsigned int iHit = 1; iHit < nHits; ++iHit) {
      int count = 0;
      if (pruneHit[iHit - 1] > 0) {
        validSet.erase(validSet.begin() + pruneHit[iHit - 1] - count);
        ++count;
      }
    }
  }
}

std::vector<SeedCandidate> SETSeedFinder::fillSeedCandidates(std::vector<MuonRecHitContainer>& allValidSets) {
  //---- we have the valid sets constructed; transform the information in an
  //---- apropriate form; meanwhile - estimate the momentum for a given set

  // RPCs should not be used (no parametrization)
  std::vector<SeedCandidate> seedCandidates_inCluster;
  // calculate and fill the inputs needed
  // loop over all valid sets
  for (unsigned int iSet = 0; iSet < allValidSets.size(); ++iSet) {
    //
    //std::cout<<"  This is SET number : "<<iSet<<std::endl;
    //for(unsigned int iHit = 0;iHit<allValidSets[iSet].size();++iHit){
    //std::cout<<"   measurements in the SET:  iHit = "<<iHit<<" pos = "<<allValidSets[iSet][iHit]->globalPosition()<<
    //" dim = "<<allValidSets[iSet][iHit]->dimension()<<std::endl;
    //}

    CLHEP::Hep3Vector momEstimate;
    int chargeEstimate;
    estimateMomentum(allValidSets[iSet], momEstimate, chargeEstimate);
    MuonRecHitContainer MuonRecHitContainer_theSet_prep;
    // currently hardcoded - will be in proper loop of course:

    SeedCandidate seedCandidates_inCluster_prep;
    seedCandidates_inCluster_prep.theSet = allValidSets[iSet];
    seedCandidates_inCluster_prep.momentum = momEstimate;
    seedCandidates_inCluster_prep.charge = chargeEstimate;
    seedCandidates_inCluster.push_back(seedCandidates_inCluster_prep);
    // END estimateMomentum
  }
  return seedCandidates_inCluster;
}

void SETSeedFinder::estimateMomentum(const MuonRecHitContainer& validSet,
                                     CLHEP::Hep3Vector& momEstimate,
                                     int& charge) const {
  int firstMeasurement = -1;
  int lastMeasurement = -1;

  // don't use 2D measurements for momentum estimation

  //if( 4==allValidSets[iSet].front()->dimension() &&
  //(allValidSets[iSet].front()->isCSC() || allValidSets[iSet].front()->isDT())){
  //firstMeasurement = 0;
  //}
  //else{
  // which is the "first" hit (4D)?
  for (unsigned int iMeas = 0; iMeas < validSet.size(); ++iMeas) {
    if (4 == validSet[iMeas]->dimension() && (validSet[iMeas]->isCSC() || validSet[iMeas]->isDT())) {
      firstMeasurement = iMeas;
      break;
    }
  }
  //}

  std::vector<double> momentum_estimate;
  double pT = 0.;
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer firstHit;
  MuonTransientTrackingRecHit::ConstMuonRecHitPointer secondHit;
  // which is the second hit?
  for (int loop = 0; loop < 2; ++loop) {  // it is actually not used; to be removed
    // this is the last measurement
    if (!loop) {  // this is what is used currently
      // 23.04.09 : it becomes a problem with introduction of ME42 chambers -
      // the initial pT parametrization is incorrect for them
      for (int iMeas = validSet.size() - 1; iMeas > -1; --iMeas) {
        if (4 == validSet[iMeas]->dimension() && (validSet[iMeas]->isCSC() || validSet[iMeas]->isDT()) &&
            // below is a fix saying "don't use ME4 chambers for initial pT estimation";
            // not using ME41 should not be a big loss too (and is more "symmetric" solution)
            fabs(validSet[iMeas]->globalPosition().z()) < 1000.) {
          lastMeasurement = iMeas;
          break;
        }
      }
    } else {
      // this is the second measurement
      for (unsigned int iMeas = 1; iMeas < validSet.size(); ++iMeas) {
        if (4 == validSet[iMeas]->dimension() && (validSet[iMeas]->isCSC() || validSet[iMeas]->isDT())) {
          lastMeasurement = iMeas;
          break;
        }
      }
    }
    // only 2D measurements (it should have been already abandoned)
    if (-1 == lastMeasurement && -1 == firstMeasurement) {
      firstMeasurement = 0;
      lastMeasurement = validSet.size() - 1;
    }
    // because of the ME42 above lastMeasurement could be -1
    else if (-1 == lastMeasurement) {
      lastMeasurement = firstMeasurement;
    } else if (-1 == firstMeasurement) {
      firstMeasurement = lastMeasurement;
    }

    firstHit = validSet[firstMeasurement];
    secondHit = validSet[lastMeasurement];
    if (firstHit->isRPC() && secondHit->isRPC()) {  // remove all RPCs from here?
      momentum_estimate.push_back(300.);
      momentum_estimate.push_back(300.);
    } else {
      if (firstHit->isRPC()) {
        firstHit = secondHit;
      } else if (secondHit->isRPC()) {
        secondHit = firstHit;
      }
      //---- estimate pT given two hits
      //std::cout<<"   hits for initial pT estimate: first -> dim = "<<firstHit->dimension()<<" pos = "<<firstHit->globalPosition()<<
      //" , second -> "<<" dim = "<<secondHit->dimension()<<" pos = "<<secondHit->globalPosition()<<std::endl;
      //---- pT throws exception if hits are MB4
      // (no coding for them - 2D hits in the outer station)
      if (2 == firstHit->dimension() && 2 == secondHit->dimension()) {
        momentum_estimate.push_back(999999999.);
        momentum_estimate.push_back(999999999.);
      } else {
        momentum_estimate = thePtExtractor->pT_extract(firstHit, secondHit);
      }
    }
    pT = fabs(momentum_estimate[0]);
    if (true || pT > 40.) {  //it is skipped; we have to look at least into number of hits in the chamber actually...
      // and then decide which segment to use
      // use the last measurement, otherwise use the second; this is to be investigated
      break;
    }
  }

  const float pT_min = 1.99;  // many hardcoded - remove them!
  if (pT > 3000.) {
    pT = 3000.;
  } else if (pT < pT_min) {
    if (pT > 0) {
      pT = pT_min;
    } else if (pT > (-1) * pT_min) {
      pT = (-1) * pT_min;
    } else if (pT < -3000.) {
      pT = -3000;
    }
  }
  //std::cout<<"  THE pT from the parametrization: "<<momentum_estimate[0]<<std::endl;
  // estimate the charge of the track candidate from the delta phi of two segments:
  //int charge      = dPhi > 0 ? 1 : -1; // what we want is: dphi < 0 => charge = -1
  charge = momentum_estimate[0] > 0 ? 1 : -1;

  // we have the pT - get the 3D momentum estimate as well

  // this is already final info:
  double xHit = validSet[firstMeasurement]->globalPosition().x();
  double yHit = validSet[firstMeasurement]->globalPosition().y();
  double rHit = TMath::Sqrt(pow(xHit, 2) + pow(yHit, 2));

  double thetaInner = validSet[firstMeasurement]->globalPosition().theta();
  // if some of the segments is missing r-phi measurement then we should
  // use only the 4D phi estimate (also use 4D eta estimate only)
  // the direction is not so important (it will be corrected)

  double rTrack = (pT / (0.3 * 3.8)) * 100.;  //times 100 for conversion to cm!

  double par = -1. * (2. / charge) * (TMath::ASin(rHit / (2 * rTrack)));
  double sinPar = TMath::Sin(par);
  double cosPar = TMath::Cos(par);

  // calculate phi at coordinate origin (0,0,0).
  double sinPhiH = 1. / (2. * charge * rTrack) * (xHit + ((sinPar) / (cosPar - 1.)) * yHit);
  double cosPhiH = -1. / (2. * charge * rTrack) * (((sinPar) / (1. - cosPar)) * xHit + yHit);

  // finally set the return vector

  // try out the reco info:
  // should used into to theta directly here (rather than tan(atan2(...)))
  momEstimate = CLHEP::Hep3Vector(pT * cosPhiH, pT * sinPhiH, pT / TMath::Tan(thetaInner));
  //Hep3Vector momEstimate(6.97961,      5.89732,     -50.0855);
  const float minMomenum = 5.;  //hardcoded - remove it! same in SETFilter
  if (momEstimate.mag() < minMomenum) {
    int sign = (pT < 0.) ? -1 : 1;
    pT = sign * (fabs(pT) + 1);
    CLHEP::Hep3Vector momEstimate2(pT * cosPhiH, pT * sinPhiH, pT / TMath::Tan(thetaInner));
    momEstimate = momEstimate2;
    if (momEstimate.mag() < minMomenum) {
      pT = sign * (fabs(pT) + 1);
      CLHEP::Hep3Vector momEstimate3(pT * cosPhiH, pT * sinPhiH, pT / TMath::Tan(thetaInner));
      momEstimate = momEstimate3;
      if (momEstimate.mag() < minMomenum) {
        pT = sign * (fabs(pT) + 1);
        CLHEP::Hep3Vector momEstimate4(pT * cosPhiH, pT * sinPhiH, pT / TMath::Tan(thetaInner));
        momEstimate = momEstimate4;
      }
    }
  }
}

TrajectorySeed SETSeedFinder::makeSeed(const TrajectoryStateOnSurface& firstTSOS,
                                       const TransientTrackingRecHit::ConstRecHitContainer& hits) const {
  edm::OwnVector<TrackingRecHit> recHitsContainer;
  for (unsigned int iHit = 0; iHit < hits.size(); ++iHit) {
    recHitsContainer.push_back(hits.at(iHit)->hit()->clone());
  }
  PropagationDirection dir = oppositeToMomentum;
  if (useSegmentsInTrajectory) {
    dir = alongMomentum;  // why forward (for rechits) later?
  }

  PTrajectoryStateOnDet const& seedTSOS =
      trajectoryStateTransform::persistentState(firstTSOS, hits.at(0)->geographicalId().rawId());
  TrajectorySeed seed(seedTSOS, recHitsContainer, dir);

  //MuonPatternRecoDumper debug;
  //std::cout<<" firstTSOS = "<<debug.dumpTSOS(firstTSOS)<<std::endl;
  //std::cout<<" iTraj = ???"<<" hits = "<<range.second-range.first<<std::endl;
  //std::cout<<" nhits = "<<hits.size()<<std::endl;
  //for(unsigned int iRH=0;iRH<hits.size();++iRH){
  //std::cout<<" RH = "<<iRH+1<<" globPos = "<<hits.at(iRH)->globalPosition()<<std::endl;
  //}
  return seed;
}
