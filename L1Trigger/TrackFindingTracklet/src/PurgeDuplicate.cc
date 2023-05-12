#include "L1Trigger/TrackFindingTracklet/interface/PurgeDuplicate.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/CleanTrackMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"

#ifdef USEHYBRID
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTracklet/interface/HybridFit.h"
#endif

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <unordered_set>
#include <algorithm>

using namespace std;
using namespace trklet;

PurgeDuplicate::PurgeDuplicate(std::string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global) {}

void PurgeDuplicate::addOutput(MemoryBase* memory, std::string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  unordered_set<string> outputs = {"trackout",
                                   "trackout1",
                                   "trackout2",
                                   "trackout3",
                                   "trackout4",
                                   "trackout5",
                                   "trackout6",
                                   "trackout7",
                                   "trackout8",
                                   "trackout9",
                                   "trackout10",
                                   "trackout11"};
  if (outputs.find(output) != outputs.end()) {
    auto* tmp = dynamic_cast<CleanTrackMemory*>(memory);
    assert(tmp != nullptr);
    outputtracklets_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find output: " << output;
}

void PurgeDuplicate::addInput(MemoryBase* memory, std::string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  unordered_set<string> inputs = {"trackin",
                                  "trackin1",
                                  "trackin2",
                                  "trackin3",
                                  "trackin4",
                                  "trackin5",
                                  "trackin6",
                                  "trackin7",
                                  "trackin8",
                                  "trackin9",
                                  "trackin10",
                                  "trackin11",
                                  "trackin12"};
  if (inputs.find(input) != inputs.end()) {
    auto* tmp = dynamic_cast<TrackFitMemory*>(memory);
    assert(tmp != nullptr);
    inputtrackfits_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find input: " << input;
}

void PurgeDuplicate::execute(std::vector<Track>& outputtracks_, unsigned int iSector) {
  inputtracklets_.clear();
  inputtracks_.clear();

  inputstubidslists_.clear();
  inputstublists_.clear();
  mergedstubidslists_.clear();

  if (settings_.removalType() != "merge") {
    for (auto& inputtrackfit : inputtrackfits_) {
      if (inputtrackfit->nTracks() == 0)
        continue;
      for (unsigned int j = 0; j < inputtrackfit->nTracks(); j++) {
        Track* aTrack = inputtrackfit->getTrack(j)->getTrack();
        aTrack->setSector(iSector);
        inputtracks_.push_back(aTrack);
      }
    }
    if (inputtracks_.empty())
      return;
  }

  unsigned int numTrk = inputtracks_.size();

  ////////////////////
  // Hybrid Removal //
  ////////////////////
#ifdef USEHYBRID

  if (settings_.removalType() == "merge") {
    std::vector<std::pair<int, bool>> trackInfo;  // Track seed & duplicate flag
    // Vector to store the relative rank of the track candidate for merging, based on seed type
    std::vector<int> seedRank;

    // Get vectors from TrackFit and save them
    // inputtracklets: Tracklet objects from the FitTrack (not actually fit yet)
    // inputstublists: L1Stubs for that track
    // inputstubidslists: Stub stubIDs for that 3rack
    // mergedstubidslists: the same as inputstubidslists, but will be used during duplicate removal
    for (unsigned int i = 0; i < inputtrackfits_.size(); i++) {
      if (inputtrackfits_[i]->nStublists() == 0)
        continue;
      if (inputtrackfits_[i]->nStublists() != inputtrackfits_[i]->nTracks())
        throw "Number of stublists and tracks don't match up!";
      for (unsigned int j = 0; j < inputtrackfits_[i]->nStublists(); j++) {
        Tracklet* aTrack = inputtrackfits_[i]->getTrack(j);
        inputtracklets_.push_back(inputtrackfits_[i]->getTrack(j));

        std::vector<const Stub*> stublist = inputtrackfits_[i]->getStublist(j);

        inputstublists_.push_back(stublist);

        std::vector<std::pair<int, int>> stubidslist = inputtrackfits_[i]->getStubidslist(j);
        inputstubidslists_.push_back(stubidslist);
        mergedstubidslists_.push_back(stubidslist);

        // Encoding: L1L2=0, L2L3=1, L3L4=2, L5L6=3, D1D2=4, D3D4=5, L1D1=6, L2D1=7
        // Best Guess:          L1L2 > L1D1 > L2L3 > L2D1 > D1D2 > L3L4 > L5L6 > D3D4
        // Best Rank:           L1L2 > L3L4 > D3D4 > D1D2 > L2L3 > L2D1 > L5L6 > L1D1
        // Rank-Informed Guess: L1L2 > L3L4 > L1D1 > L2L3 > L2D1 > D1D2 > L5L6 > D3D4
        unsigned int curSeed = aTrack->seedIndex();
        if (curSeed == 0) {
          seedRank.push_back(1);
        } else if (curSeed == 2) {
          seedRank.push_back(2);
        } else if (curSeed == 5) {
          seedRank.push_back(3);
        } else if (curSeed == 4) {
          seedRank.push_back(4);
        } else if (curSeed == 1) {
          seedRank.push_back(5);
        } else if (curSeed == 7) {
          seedRank.push_back(6);
        } else if (curSeed == 3) {
          seedRank.push_back(7);
        } else if (curSeed == 6) {
          seedRank.push_back(8);
        } else if (settings_.extended()) {
          seedRank.push_back(9);
        } else {
          throw cms::Exception("LogError") << __FILE__ << " " << __LINE__ << " Seed " << curSeed
                                           << " not found in list, and settings->extended() not set.";
        }

        if (stublist.size() != stubidslist.size())
          throw "Number of stubs and stubids don't match up!";

        trackInfo.emplace_back(i, false);
      }
    }

    if (inputtracklets_.empty())
      return;
    unsigned int numStublists = inputstublists_.size();

    // Initialize all-false 2D array of tracks being duplicates to other tracks
    bool dupMap[numStublists][numStublists];  // Ends up symmetric
    for (unsigned int itrk = 0; itrk < numStublists; itrk++) {
      for (unsigned int jtrk = 0; jtrk < numStublists; jtrk++) {
        dupMap[itrk][jtrk] = false;
      }
    }

    // Find duplicates; Fill dupMap by looping over all pairs of "tracks"
    // numStublists-1 since last track has no other to compare to
    for (unsigned int itrk = 0; itrk < numStublists - 1; itrk++) {
      for (unsigned int jtrk = itrk + 1; jtrk < numStublists; jtrk++) {
        // Get primary track stubids
        const std::vector<std::pair<int, int>>& stubsTrk1 = inputstubidslists_[itrk];

        // Get and count secondary track stubids
        const std::vector<std::pair<int, int>>& stubsTrk2 = inputstubidslists_[jtrk];

        // Count number of Unique Regions (UR) that share stubs, and the number of UR that each track hits
        unsigned int nShareUR = 0;
        if (settings_.mergeComparison() == "CompareAll") {
          bool URArray[16];
          for (auto& i : URArray) {
            i = false;
          };
          for (const auto& st1 : stubsTrk1) {
            for (const auto& st2 : stubsTrk2) {
              if (st1.first == st2.first && st1.second == st2.second) {
                // Converts region encoded in st1->first to an index in the Unique Region (UR) array
                int i = st1.first;
                int reg = (i > 0 && i < 10) * (i - 1) + (i > 10) * (i - 5) - (i < 0) * i;
                if (!URArray[reg]) {
                  nShareUR++;
                  URArray[reg] = true;
                }
              }
            }
          }
        } else if (settings_.mergeComparison() == "CompareBest") {
          std::vector<const Stub*> fullStubslistsTrk1 = inputstublists_[itrk];
          std::vector<const Stub*> fullStubslistsTrk2 = inputstublists_[jtrk];

          // Arrays to store the index of the best stub in each region
          int URStubidsTrk1[16];
          int URStubidsTrk2[16];
          for (int i = 0; i < 16; i++) {
            URStubidsTrk1[i] = -1;
            URStubidsTrk2[i] = -1;
          }
          // For each stub on the first track, find the stub with the best residual and store its index in the URStubidsTrk1 array
          for (unsigned int stcount = 0; stcount < stubsTrk1.size(); stcount++) {
            int i = stubsTrk1[stcount].first;
            int reg = (i > 0 && i < 10) * (i - 1) + (i > 10) * (i - 5) - (i < 0) * i;
            double nres = getPhiRes(inputtracklets_[itrk], fullStubslistsTrk1[stcount]);
            double ores = 0;
            if (URStubidsTrk1[reg] != -1)
              ores = getPhiRes(inputtracklets_[itrk], fullStubslistsTrk1[URStubidsTrk1[reg]]);
            if (URStubidsTrk1[reg] == -1 || nres < ores) {
              URStubidsTrk1[reg] = stcount;
            }
          }
          // For each stub on the second track, find the stub with the best residual and store its index in the URStubidsTrk1 array
          for (unsigned int stcount = 0; stcount < stubsTrk2.size(); stcount++) {
            int i = stubsTrk2[stcount].first;
            int reg = (i > 0 && i < 10) * (i - 1) + (i > 10) * (i - 5) - (i < 0) * i;
            double nres = getPhiRes(inputtracklets_[jtrk], fullStubslistsTrk2[stcount]);
            double ores = 0;
            if (URStubidsTrk2[reg] != -1)
              ores = getPhiRes(inputtracklets_[jtrk], fullStubslistsTrk2[URStubidsTrk2[reg]]);
            if (URStubidsTrk2[reg] == -1 || nres < ores) {
              URStubidsTrk2[reg] = stcount;
            }
          }
          // For all 16 regions (6 layers and 10 disks), count the number of regions who's best stub on both tracks are the same
          for (int i = 0; i < 16; i++) {
            int t1i = URStubidsTrk1[i];
            int t2i = URStubidsTrk2[i];
            if (t1i != -1 && t2i != -1 && stubsTrk1[t1i].first == stubsTrk2[t2i].first &&
                stubsTrk1[t1i].second == stubsTrk2[t2i].second)
              nShareUR++;
          }
        }

        // Fill duplicate map
        if (nShareUR >= settings_.minIndStubs()) {  // For number of shared stub merge condition
          dupMap[itrk][jtrk] = true;
          dupMap[jtrk][itrk] = true;
        }
      }
    }

    // Merge duplicate tracks
    for (unsigned int itrk = 0; itrk < numStublists - 1; itrk++) {
      for (unsigned int jtrk = itrk + 1; jtrk < numStublists; jtrk++) {
        // Merge a track with its first duplicate found.
        if (dupMap[itrk][jtrk]) {
          // Set preferred track based on seed rank
          int preftrk;
          int rejetrk;
          if (seedRank[itrk] < seedRank[jtrk]) {
            preftrk = itrk;
            rejetrk = jtrk;
          } else {
            preftrk = jtrk;
            rejetrk = itrk;
          }

          // Get a merged stub list
          std::vector<const Stub*> newStubList;
          std::vector<const Stub*> stubsTrk1 = inputstublists_[rejetrk];
          std::vector<const Stub*> stubsTrk2 = inputstublists_[preftrk];
          newStubList = stubsTrk1;
          for (unsigned int stub2it = 0; stub2it < stubsTrk2.size(); stub2it++) {
            if (find(stubsTrk1.begin(), stubsTrk1.end(), stubsTrk2[stub2it]) == stubsTrk1.end()) {
              newStubList.push_back(stubsTrk2[stub2it]);
            }
          }
          // Overwrite stublist of preferred track with merged list
          inputstublists_[preftrk] = newStubList;

          std::vector<std::pair<int, int>> newStubidsList;
          std::vector<std::pair<int, int>> stubidsTrk1 = mergedstubidslists_[rejetrk];
          std::vector<std::pair<int, int>> stubidsTrk2 = mergedstubidslists_[preftrk];
          newStubidsList = stubidsTrk1;
          for (unsigned int stub2it = 0; stub2it < stubidsTrk2.size(); stub2it++) {
            if (find(stubidsTrk1.begin(), stubidsTrk1.end(), stubidsTrk2[stub2it]) == stubidsTrk1.end()) {
              newStubidsList.push_back(stubidsTrk2[stub2it]);
            }
          }
          // Overwrite stubidslist of preferred track with merged list
          mergedstubidslists_[preftrk] = newStubidsList;

          // Mark that rejected track has been merged into another track
          trackInfo[rejetrk].second = true;
        }
      }
    }

    // Make the final track objects, fit with KF, and send to output
    for (unsigned int itrk = 0; itrk < numStublists; itrk++) {
      bool duplicateTrack = trackInfo[itrk].second;
      if (not duplicateTrack) {  // Don't waste CPU by calling KF for duplicates

        Tracklet* tracklet = inputtracklets_[itrk];
        std::vector<const Stub*> trackstublist = inputstublists_[itrk];

        // Run KF track fit
        HybridFit hybridFitter(iSector, settings_, globals_);
        hybridFitter.Fit(tracklet, trackstublist);

        // If the track was accepted (and thus fit), add to output
        if (tracklet->fit()) {
          // Add fitted Track to output (later converted to TTTrack)
          Track* outtrack = tracklet->getTrack();
          outtrack->setSector(iSector);
          // Also store fitted track as more detailed Tracklet object.
          outputtracklets_[trackInfo[itrk].first]->addTrack(tracklet);

          // Add all tracks to standalone root file output
          outtrack->setStubIDpremerge(inputstubidslists_[itrk]);
          outtrack->setStubIDprefit(mergedstubidslists_[itrk]);
          outputtracks_.push_back(*outtrack);
        }
      }
    }
  }
#endif

  //////////////////
  // Grid removal //
  //////////////////
  if (settings_.removalType() == "grid") {
    // Sort tracks by ichisq/DoF so that removal will keep the lower ichisq/DoF track
    std::sort(inputtracks_.begin(), inputtracks_.end(), [](const Track* lhs, const Track* rhs) {
      return lhs->ichisq() / lhs->stubID().size() < rhs->ichisq() / rhs->stubID().size();
    });
    bool grid[35][40] = {{false}};

    for (unsigned int itrk = 0; itrk < numTrk; itrk++) {
      if (inputtracks_[itrk]->duplicate())
        edm::LogPrint("Tracklet") << "WARNING: Track already tagged as duplicate!!";

      double phiBin = (inputtracks_[itrk]->phi0(settings_) - 2 * M_PI / 27 * iSector) / (2 * M_PI / 9 / 50) + 9;
      phiBin = std::max(phiBin, 0.);
      phiBin = std::min(phiBin, 34.);

      double ptBin = 1 / inputtracks_[itrk]->pt(settings_) * 40 + 20;
      ptBin = std::max(ptBin, 0.);
      ptBin = std::min(ptBin, 39.);

      if (grid[(int)phiBin][(int)ptBin])
        inputtracks_[itrk]->setDuplicate(true);
      grid[(int)phiBin][(int)ptBin] = true;

      double phiTest = inputtracks_[itrk]->phi0(settings_) - 2 * M_PI / 27 * iSector;
      if (phiTest < -2 * M_PI / 27)
        edm::LogVerbatim("Tracklet") << "track phi too small!";
      if (phiTest > 2 * 2 * M_PI / 27)
        edm::LogVerbatim("Tracklet") << "track phi too big!";
    }
  }  // end grid removal

  //////////////////////////
  // ichi + nstub removal //
  //////////////////////////
  if (settings_.removalType() == "ichi" || settings_.removalType() == "nstub") {
    for (unsigned int itrk = 0; itrk < numTrk - 1; itrk++) {  // numTrk-1 since last track has no other to compare to

      // If primary track is a duplicate, it cannot veto any...move on
      if (inputtracks_[itrk]->duplicate() == 1)
        continue;

      unsigned int nStubP = 0;
      vector<unsigned int> nStubS(numTrk);
      vector<unsigned int> nShare(numTrk);
      // Get and count primary stubs
      std::map<int, int> stubsTrk1 = inputtracks_[itrk]->stubID();
      nStubP = stubsTrk1.size();

      for (unsigned int jtrk = itrk + 1; jtrk < numTrk; jtrk++) {
        // Skip duplicate tracks
        if (inputtracks_[jtrk]->duplicate() == 1)
          continue;

        // Get and count secondary stubs
        std::map<int, int> stubsTrk2 = inputtracks_[jtrk]->stubID();
        nStubS[jtrk] = stubsTrk2.size();

        // Count shared stubs
        for (auto& st : stubsTrk1) {
          if (stubsTrk2.find(st.first) != stubsTrk2.end()) {
            if (st.second == stubsTrk2[st.first])
              nShare[jtrk]++;
          }
        }
      }

      // Tag duplicates
      for (unsigned int jtrk = itrk + 1; jtrk < numTrk; jtrk++) {
        // Skip duplicate tracks
        if (inputtracks_[jtrk]->duplicate() == 1)
          continue;

        // Chi2 duplicate removal
        if (settings_.removalType() == "ichi") {
          if ((nStubP - nShare[jtrk] < settings_.minIndStubs()) ||
              (nStubS[jtrk] - nShare[jtrk] < settings_.minIndStubs())) {
            if ((int)inputtracks_[itrk]->ichisq() / (2 * inputtracks_[itrk]->stubID().size() - 4) >
                (int)inputtracks_[jtrk]->ichisq() / (2 * inputtracks_[itrk]->stubID().size() - 4)) {
              inputtracks_[itrk]->setDuplicate(true);
            } else if ((int)inputtracks_[itrk]->ichisq() / (2 * inputtracks_[itrk]->stubID().size() - 4) <=
                       (int)inputtracks_[jtrk]->ichisq() / (2 * inputtracks_[itrk]->stubID().size() - 4)) {
              inputtracks_[jtrk]->setDuplicate(true);
            } else {
              edm::LogVerbatim("Tracklet") << "Error: Didn't tag either track in duplicate pair.";
            }
          }
        }  // end ichi removal

        // nStub duplicate removal
        if (settings_.removalType() == "nstub") {
          if ((nStubP - nShare[jtrk] < settings_.minIndStubs()) && (nStubP < nStubS[jtrk])) {
            inputtracks_[itrk]->setDuplicate(true);
          } else if ((nStubS[jtrk] - nShare[jtrk] < settings_.minIndStubs()) && (nStubS[jtrk] <= nStubP)) {
            inputtracks_[jtrk]->setDuplicate(true);
          } else {
            edm::LogVerbatim("Tracklet") << "Error: Didn't tag either track in duplicate pair.";
          }
        }  // end nstub removal

      }  // end tag duplicates

    }  // end loop over primary track

  }  // end ichi + nstub removal

  //Add tracks to output
  if (settings_.removalType() != "merge") {
    for (unsigned int i = 0; i < inputtrackfits_.size(); i++) {
      for (unsigned int j = 0; j < inputtrackfits_[i]->nTracks(); j++) {
        if (inputtrackfits_[i]->getTrack(j)->getTrack()->duplicate() == 0) {
          if (settings_.writeMonitorData("Seeds")) {
            ofstream fout("seeds.txt", ofstream::app);
            fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector << " "
                 << inputtrackfits_[i]->getTrack(j)->getISeed() << endl;
            fout.close();
          }
          outputtracklets_[i]->addTrack(inputtrackfits_[i]->getTrack(j));
        }
        //For root file:
        outputtracks_.push_back(*inputtrackfits_[i]->getTrack(j)->getTrack());
      }
    }
  }
}

double PurgeDuplicate::getPhiRes(Tracklet* curTracklet, const Stub* curStub) {
  double phiproj;
  double stubphi;
  double phires;
  // Get phi position of stub
  stubphi = curStub->l1tstub()->phi();
  // Get region that the stub is in (Layer 1->6, Disk 1->5)
  int Layer = curStub->layerdisk() + 1;
  if (Layer > N_LAYER) {
    Layer = 0;
  }
  int Disk = curStub->layerdisk() - (N_LAYER - 1);
  if (Disk < 0) {
    Disk = 0;
  }
  // Get phi projection of tracklet
  int seedindex = curTracklet->seedIndex();
  // If this stub is a seed stub, set projection=phi, so that res=0
  if ((seedindex == 0 && (Layer == 1 || Layer == 2)) || (seedindex == 1 && (Layer == 2 || Layer == 3)) ||
      (seedindex == 2 && (Layer == 3 || Layer == 4)) || (seedindex == 3 && (Layer == 5 || Layer == 6)) ||
      (seedindex == 4 && (abs(Disk) == 1 || abs(Disk) == 2)) ||
      (seedindex == 5 && (abs(Disk) == 3 || abs(Disk) == 4)) || (seedindex == 6 && (Layer == 1 || abs(Disk) == 1)) ||
      (seedindex == 7 && (Layer == 2 || abs(Disk) == 1)) ||
      (seedindex == 8 && (Layer == 2 || Layer == 3 || Layer == 4)) ||
      (seedindex == 9 && (Layer == 4 || Layer == 5 || Layer == 6)) ||
      (seedindex == 10 && (Layer == 2 || Layer == 3 || abs(Disk) == 1)) ||
      (seedindex == 11 && (Layer == 2 || abs(Disk) == 1 || abs(Disk) == 2))) {
    phiproj = stubphi;
    // Otherwise, get projection of tracklet
  } else {
    phiproj = curTracklet->proj(curStub->layerdisk()).phiproj();
  }
  // Calculate residual
  phires = std::abs(stubphi - phiproj);
  return phires;
}
