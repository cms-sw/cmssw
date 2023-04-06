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
    // Track seed & duplicate flag
    std::vector<std::pair<int, bool>> trackInfo;
    // Flag for tracks in multiple bins that get merged but are not in the correct variable bin
    std::vector<bool> trackBinInfo;
    // Vector to store the relative rank of the track candidate for merging, based on seed type
    std::vector<int> seedRank;

    // Stubs on every track
    std::vector<std::vector<const Stub*>> inputstublistsall;
    // (layer, unique stub index within layer) of each stub on every track
    std::vector<std::vector<std::pair<int, int>>> mergedstubidslistsall;
    std::vector<std::vector<std::pair<int, int>>> inputstubidslistsall;
    std::vector<Tracklet*> inputtrackletsall;

    std::vector<unsigned int> prefTracks;  // Stores all the tracks that are sent to the KF from each bin
    std::vector<int> prefTrackFit;  // Stores the track seed that corresponds to the associated track in prefTracks

    for (unsigned int bin = 0; bin < settings_.varRInvBins().size() - 1; bin++) {
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
          if (isTrackInBin(findOverlapRInvBins(inputtrackfits_[i]->getTrack(j)), bin)) {
            if (inputtracklets_.size() >= settings_.maxStep("DR"))
              continue;
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
            std::vector<int> ranks{1, 5, 2, 7, 4, 3, 8, 6};
            if (settings_.extended())
              seedRank.push_back(9);
            else
              seedRank.push_back(ranks[curSeed]);

            if (stublist.size() != stubidslist.size())
              throw "Number of stubs and stubids don't match up!";

            trackInfo.emplace_back(i, false);
            trackBinInfo.emplace_back(false);
          } else
            continue;
        }
      }

      if (inputtracklets_.empty())
        continue;
      const unsigned int numStublists = inputstublists_.size();

      if (settings_.inventStubs()) {
        for (unsigned int itrk = 0; itrk < numStublists; itrk++) {
          inputstublists_[itrk] = getInventedSeedingStub(iSector, inputtracklets_[itrk], inputstublists_[itrk]);
        }
      }

      // Initialize all-false 2D array of tracks being duplicates to other tracks
      bool dupMap[numStublists][numStublists];  // Ends up symmetric
      for (unsigned int itrk = 0; itrk < numStublists; itrk++) {
        for (unsigned int jtrk = 0; jtrk < numStublists; jtrk++) {
          dupMap[itrk][jtrk] = false;
        }
      }

      // Used to check if a track is in two bins, is not a duplicate in either bin, so is sent out twice
      bool noMerge[numStublists];
      for (unsigned int itrk = 0; itrk < numStublists; itrk++) {
        noMerge[itrk] = false;
      }

      // Find duplicates; Fill dupMap by looping over all pairs of "tracks"
      // numStublists-1 since last track has no other to compare to
      for (unsigned int itrk = 0; itrk < numStublists - 1; itrk++) {
        for (unsigned int jtrk = itrk + 1; jtrk < numStublists; jtrk++) {
          if (itrk >= settings_.numTracksComparedPerBin())
            continue;
          // Get primary track stubids = (layer, unique stub index within layer)
          const std::vector<std::pair<int, int>>& stubsTrk1 = inputstubidslists_[itrk];

          // Get and count secondary track stubids
          const std::vector<std::pair<int, int>>& stubsTrk2 = inputstubidslists_[jtrk];

          // Count number of layers that share stubs, and the number of UR that each track hits
          unsigned int nShareLay = 0;
          unsigned int nLayStubTrk1 = 0;
          unsigned int nLayStubTrk2 = 0;
          if (settings_.mergeComparison() == "CompareAll") {
            bool layerArr[16];
            for (auto& i : layerArr) {
              i = false;
            };
            for (const auto& st1 : stubsTrk1) {
              for (const auto& st2 : stubsTrk2) {
                if (st1.first == st2.first && st1.second == st2.second) {  // tracks share stub
                  // Converts layer/disk encoded in st1->first to an index in the layer array
                  int i = st1.first;  // layer/disk
                  bool barrel = (i > 0 && i < 10);
                  bool endcapA = (i > 10);
                  bool endcapB = (i < 0);
                  int lay = barrel * (i - 1) + endcapA * (i - 5) - endcapB * i;  // encode in range 0-15
                  if (!layerArr[lay]) {
                    nShareLay++;
                    layerArr[lay] = true;
                  }
                }
              }
            }
          } else if (settings_.mergeComparison() == "CompareBest") {
            std::vector<const Stub*> fullStubslistsTrk1 = inputstublists_[itrk];
            std::vector<const Stub*> fullStubslistsTrk2 = inputstublists_[jtrk];

            // Arrays to store the index of the best stub in each layer
            int layStubidsTrk1[16];
            int layStubidsTrk2[16];
            for (int i = 0; i < 16; i++) {
              layStubidsTrk1[i] = -1;
              layStubidsTrk2[i] = -1;
            }
            // For each stub on the first track, find the stub with the best residual and store its index in the layStubidsTrk1 array
            for (unsigned int stcount = 0; stcount < stubsTrk1.size(); stcount++) {
              int i = stubsTrk1[stcount].first;  // layer/disk
              bool barrel = (i > 0 && i < 10);
              bool endcapA = (i > 10);
              bool endcapB = (i < 0);
              int lay = barrel * (i - 1) + endcapA * (i - 5) - endcapB * i;  // encode in range 0-15
              double nres = getPhiRes(inputtracklets_[itrk], fullStubslistsTrk1[stcount]);
              double ores = 0;
              if (layStubidsTrk1[lay] != -1)
                ores = getPhiRes(inputtracklets_[itrk], fullStubslistsTrk1[layStubidsTrk1[lay]]);
              if (layStubidsTrk1[lay] == -1 || nres < ores) {
                layStubidsTrk1[lay] = stcount;
              }
            }
            // For each stub on the second track, find the stub with the best residual and store its index in the layStubidsTrk1 array
            for (unsigned int stcount = 0; stcount < stubsTrk2.size(); stcount++) {
              int i = stubsTrk2[stcount].first;  // layer/disk
              bool barrel = (i > 0 && i < 10);
              bool endcapA = (i > 10);
              bool endcapB = (i < 0);
              int lay = barrel * (i - 1) + endcapA * (i - 5) - endcapB * i;  // encode in range 0-15
              double nres = getPhiRes(inputtracklets_[jtrk], fullStubslistsTrk2[stcount]);
              double ores = 0;
              if (layStubidsTrk2[lay] != -1)
                ores = getPhiRes(inputtracklets_[jtrk], fullStubslistsTrk2[layStubidsTrk2[lay]]);
              if (layStubidsTrk2[lay] == -1 || nres < ores) {
                layStubidsTrk2[lay] = stcount;
              }
            }
            // For all 16 layers (6 layers and 10 disks), count the number of layers who's best stub on both tracks are the same
            for (int i = 0; i < 16; i++) {
              int t1i = layStubidsTrk1[i];
              int t2i = layStubidsTrk2[i];
              if (t1i != -1 && t2i != -1 && stubsTrk1[t1i].first == stubsTrk2[t2i].first &&
                  stubsTrk1[t1i].second == stubsTrk2[t2i].second)
                nShareLay++;
            }
            // Calculate the number of layers hit by each track, so that this number can be used in calculating the number of independent
            // stubs on a track (not enabled/used by default)
            for (int i = 0; i < 16; i++) {
              if (layStubidsTrk1[i] != -1)
                nLayStubTrk1++;
              if (layStubidsTrk2[i] != -1)
                nLayStubTrk2++;
            }
          }

          // Fill duplicate map
          if (nShareLay >= settings_.minIndStubs()) {  // For number of shared stub merge condition
            dupMap[itrk][jtrk] = true;
            dupMap[jtrk][itrk] = true;
          }
        }
      }

      // Check to see if the track is a duplicate
      for (unsigned int itrk = 0; itrk < numStublists; itrk++) {
        for (unsigned int jtrk = 0; jtrk < numStublists; jtrk++) {
          if (dupMap[itrk][jtrk]) {
            noMerge[itrk] = true;
          }
        }
      }

      // If the track isn't a duplicate, and if it's in more than one bin, and it is not in the proper varrinvbin, then mark it so it won't be sent to output
      for (unsigned int itrk = 0; itrk < numStublists; itrk++) {
        if (noMerge[itrk] == false) {
          if ((findOverlapRInvBins(inputtracklets_[itrk]).size() > 1) &&
              (findVarRInvBin(inputtracklets_[itrk]) != bin)) {
            trackInfo[itrk].second = true;
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

            // If the preffered track is in more than one bin, but not in the proper varrinvbin, then mark as true
            if ((findOverlapRInvBins(inputtracklets_[preftrk]).size() > 1) &&
                (findVarRInvBin(inputtracklets_[preftrk]) != bin)) {
              trackBinInfo[preftrk] = true;
              trackBinInfo[rejetrk] = true;
            } else {
              // Get a merged stub list
              std::vector<const Stub*> newStubList;
              std::vector<const Stub*> stubsTrk1 = inputstublists_[preftrk];
              std::vector<const Stub*> stubsTrk2 = inputstublists_[rejetrk];
              std::vector<unsigned int> stubsTrk1indices;
              std::vector<unsigned int> stubsTrk2indices;
              for (unsigned int stub1it = 0; stub1it < stubsTrk1.size(); stub1it++) {
                stubsTrk1indices.push_back(stubsTrk1[stub1it]->l1tstub()->uniqueIndex());
              }
              for (unsigned int stub2it = 0; stub2it < stubsTrk2.size(); stub2it++) {
                stubsTrk2indices.push_back(stubsTrk2[stub2it]->l1tstub()->uniqueIndex());
              }
              newStubList = stubsTrk1;
              for (unsigned int stub2it = 0; stub2it < stubsTrk2.size(); stub2it++) {
                if (find(stubsTrk1indices.begin(), stubsTrk1indices.end(), stubsTrk2indices[stub2it]) ==
                    stubsTrk1indices.end()) {
                  newStubList.push_back(stubsTrk2[stub2it]);
                }
              }
              //   Overwrite stublist of preferred track with merged list
              inputstublists_[preftrk] = newStubList;

              std::vector<std::pair<int, int>> newStubidsList;
              std::vector<std::pair<int, int>> stubidsTrk1 = mergedstubidslists_[preftrk];
              std::vector<std::pair<int, int>> stubidsTrk2 = mergedstubidslists_[rejetrk];
              newStubidsList = stubidsTrk1;

              for (unsigned int stub2it = 0; stub2it < stubsTrk2.size(); stub2it++) {
                if (find(stubsTrk1indices.begin(), stubsTrk1indices.end(), stubsTrk2indices[stub2it]) ==
                    stubsTrk1indices.end()) {
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
      }

      for (unsigned int ktrk = 0; ktrk < numStublists; ktrk++) {
        if ((trackInfo[ktrk].second != true) && (trackBinInfo[ktrk] != true)) {
          prefTracks.push_back(ktrk);
          prefTrackFit.push_back(trackInfo[ktrk].first);
          inputtrackletsall.push_back(inputtracklets_[ktrk]);
          inputstublistsall.push_back(inputstublists_[ktrk]);
          inputstubidslistsall.push_back(inputstubidslists_[ktrk]);
          mergedstubidslistsall.push_back(mergedstubidslists_[ktrk]);
        }
      }

      // Need to clear all the vectors which will be used in the next bin
      seedRank.clear();
      trackInfo.clear();
      trackBinInfo.clear();
      inputtracklets_.clear();
      inputstublists_.clear();
      inputstubidslists_.clear();
      mergedstubidslists_.clear();
    }

    // Make the final track objects, fit with KF, and send to output
    for (unsigned int itrk = 0; itrk < prefTracks.size(); itrk++) {
      Tracklet* tracklet = inputtrackletsall[itrk];
      std::vector<const Stub*> trackstublist = inputstublistsall[itrk];

      // Run KF track fit
      HybridFit hybridFitter(iSector, settings_, globals_);
      hybridFitter.Fit(tracklet, trackstublist);

      // If the track was accepted (and thus fit), add to output
      if (tracklet->fit()) {
        // Add fitted Track to output (later converted to TTTrack)
        Track* outtrack = tracklet->getTrack();
        outtrack->setSector(iSector);
        // Also store fitted track as more detailed Tracklet object.
        outputtracklets_[prefTrackFit[itrk]]->addTrack(tracklet);

        // Add all tracks to standalone root file output
        outtrack->setStubIDpremerge(inputstubidslistsall[itrk]);
        outtrack->setStubIDprefit(mergedstubidslistsall[itrk]);
        outputtracks_.push_back(*outtrack);
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

double PurgeDuplicate::getPhiRes(Tracklet* curTracklet, const Stub* curStub) const {
  double phiproj;
  double stubphi;
  double phires;
  // Get phi position of stub
  stubphi = curStub->l1tstub()->phi();
  // Get layer that the stub is in (Layer 1->6, Disk 1->5)
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
  if (isSeedingStub(seedindex, Layer, Disk)) {
    phiproj = stubphi;
    // Otherwise, get projection of tracklet
  } else {
    phiproj = curTracklet->proj(curStub->layerdisk()).phiproj();
  }
  // Calculate residual
  phires = std::abs(stubphi - phiproj);
  return phires;
}

bool PurgeDuplicate::isSeedingStub(int seedindex, int Layer, int Disk) const {
  if ((seedindex == 0 && (Layer == 1 || Layer == 2)) || (seedindex == 1 && (Layer == 2 || Layer == 3)) ||
      (seedindex == 2 && (Layer == 3 || Layer == 4)) || (seedindex == 3 && (Layer == 5 || Layer == 6)) ||
      (seedindex == 4 && (abs(Disk) == 1 || abs(Disk) == 2)) ||
      (seedindex == 5 && (abs(Disk) == 3 || abs(Disk) == 4)) || (seedindex == 6 && (Layer == 1 || abs(Disk) == 1)) ||
      (seedindex == 7 && (Layer == 2 || abs(Disk) == 1)) ||
      (seedindex == 8 && (Layer == 2 || Layer == 3 || Layer == 4)) ||
      (seedindex == 9 && (Layer == 4 || Layer == 5 || Layer == 6)) ||
      (seedindex == 10 && (Layer == 2 || Layer == 3 || abs(Disk) == 1)) ||
      (seedindex == 11 && (Layer == 2 || abs(Disk) == 1 || abs(Disk) == 2)))
    return true;

  return false;
}

std::pair<int, int> PurgeDuplicate::findLayerDisk(const Stub* st) const {
  std::pair<int, int> layer_disk;
  layer_disk.first = st->layerdisk() + 1;
  if (layer_disk.first > N_LAYER) {
    layer_disk.first = 0;
  }
  layer_disk.second = st->layerdisk() - (N_LAYER - 1);
  if (layer_disk.second < 0) {
    layer_disk.second = 0;
  }
  return layer_disk;
}

std::string PurgeDuplicate::l1tinfo(const L1TStub* l1stub, std::string str = "") const {
  std::string thestr = Form("\t %s stub info:  r/z/phi:\t%f\t%f\t%f\t%d\t%f\t%d",
                            str.c_str(),
                            l1stub->r(),
                            l1stub->z(),
                            l1stub->phi(),
                            l1stub->iphi(),
                            l1stub->bend(),
                            l1stub->layerdisk());
  return thestr;
}

std::vector<double> PurgeDuplicate::getInventedCoords(unsigned int iSector,
                                                      const Stub* st,
                                                      const Tracklet* tracklet) const {
  int stubLayer = (findLayerDisk(st)).first;
  int stubDisk = (findLayerDisk(st)).second;

  double stub_phi = -99;
  double stub_z = -99;
  double stub_r = -99;

  double tracklet_rinv = tracklet->rinv();

  if (st->isBarrel()) {
    stub_r = settings_.rmean(stubLayer - 1);
    stub_phi = tracklet->phi0() - std::asin(stub_r * tracklet_rinv / 2);
    stub_phi = stub_phi + iSector * settings_.dphisector() - 0.5 * settings_.dphisectorHG();
    stub_phi = reco::reduceRange(stub_phi);
    stub_z = tracklet->z0() + 2 * tracklet->t() * 1 / tracklet_rinv * std::asin(stub_r * tracklet_rinv / 2);
  } else {
    stub_z = settings_.zmean(stubDisk - 1) * tracklet->disk() / abs(tracklet->disk());
    stub_phi = tracklet->phi0() - (stub_z - tracklet->z0()) * tracklet_rinv / 2 / tracklet->t();
    stub_phi = stub_phi + iSector * settings_.dphisector() - 0.5 * settings_.dphisectorHG();
    stub_phi = reco::reduceRange(stub_phi);
    stub_r = 2 / tracklet_rinv * std::sin((stub_z - tracklet->z0()) * tracklet_rinv / 2 / tracklet->t());
  }

  std::vector invented_coords{stub_r, stub_z, stub_phi};
  return invented_coords;
}

std::vector<double> PurgeDuplicate::getInventedCoordsExtended(unsigned int iSector,
                                                              const Stub* st,
                                                              const Tracklet* tracklet) const {
  int stubLayer = (findLayerDisk(st)).first;
  int stubDisk = (findLayerDisk(st)).second;

  double stub_phi = -99;
  double stub_z = -99;
  double stub_r = -99;

  double rho = 1 / tracklet->rinv();
  double rho_minus_d0 = rho + tracklet->d0();  // should be -, but otherwise does not work

  // exact helix
  if (st->isBarrel()) {
    stub_r = settings_.rmean(stubLayer - 1);

    double sin_val = (stub_r * stub_r + rho_minus_d0 * rho_minus_d0 - rho * rho) / (2 * stub_r * rho_minus_d0);
    stub_phi = tracklet->phi0() - std::asin(sin_val);
    stub_phi = stub_phi + iSector * settings_.dphisector() - 0.5 * settings_.dphisectorHG();
    stub_phi = reco::reduceRange(stub_phi);

    double beta = std::acos((rho * rho + rho_minus_d0 * rho_minus_d0 - stub_r * stub_r) / (2 * rho * rho_minus_d0));
    stub_z = tracklet->z0() + tracklet->t() * std::abs(rho * beta);
  } else {
    stub_z = settings_.zmean(stubDisk - 1) * tracklet->disk() / abs(tracklet->disk());

    double beta = (stub_z - tracklet->z0()) / (tracklet->t() * std::abs(rho));  // maybe rho should be abs value
    double r_square = -2 * rho * rho_minus_d0 * std::cos(beta) + rho * rho + rho_minus_d0 * rho_minus_d0;
    stub_r = sqrt(r_square);

    double sin_val = (stub_r * stub_r + rho_minus_d0 * rho_minus_d0 - rho * rho) / (2 * stub_r * rho_minus_d0);
    stub_phi = tracklet->phi0() - std::asin(sin_val);
    stub_phi = stub_phi + iSector * settings_.dphisector() - 0.5 * settings_.dphisectorHG();
    stub_phi = reco::reduceRange(stub_phi);
  }

  // TMP: for displaced tracking, exclude one of the 3 seeding stubs
  // to be discussed
  int seed = tracklet->seedIndex();
  if ((seed == 8 && stubLayer == 4) || (seed == 9 && stubLayer == 5) || (seed == 10 && stubLayer == 3) ||
      (seed == 11 && abs(stubDisk) == 1)) {
    stub_phi = st->l1tstub()->phi();
    stub_z = st->l1tstub()->z();
    stub_r = st->l1tstub()->r();
  }

  std::vector invented_coords{stub_r, stub_z, stub_phi};
  return invented_coords;
}

std::vector<const Stub*> PurgeDuplicate::getInventedSeedingStub(
    unsigned int iSector, const Tracklet* tracklet, const std::vector<const Stub*>& originalStubsList) const {
  std::vector<const Stub*> newStubList;

  for (unsigned int stubit = 0; stubit < originalStubsList.size(); stubit++) {
    const Stub* thisStub = originalStubsList[stubit];

    if (isSeedingStub(tracklet->seedIndex(), (findLayerDisk(thisStub)).first, (findLayerDisk(thisStub)).second)) {
      // get a vector containing r, z, phi
      std::vector<double> inv_r_z_phi;
      if (!settings_.extended())
        inv_r_z_phi = getInventedCoords(iSector, thisStub, tracklet);
      else {
        inv_r_z_phi = getInventedCoordsExtended(iSector, thisStub, tracklet);
      }
      double stub_x_invent = inv_r_z_phi[0] * std::cos(inv_r_z_phi[2]);
      double stub_y_invent = inv_r_z_phi[0] * std::sin(inv_r_z_phi[2]);
      double stub_z_invent = inv_r_z_phi[1];

      Stub* invent_stub_ptr = new Stub(*thisStub);
      const L1TStub* l1stub = thisStub->l1tstub();
      L1TStub invent_l1stub = *l1stub;
      invent_l1stub.setCoords(stub_x_invent, stub_y_invent, stub_z_invent);

      invent_stub_ptr->setl1tstub(new L1TStub(invent_l1stub));
      invent_stub_ptr->l1tstub()->setAllStubIndex(l1stub->allStubIndex());
      invent_stub_ptr->l1tstub()->setUniqueIndex(l1stub->uniqueIndex());

      newStubList.push_back(invent_stub_ptr);

    } else {
      newStubList.push_back(thisStub);
    }
  }
  return newStubList;
}

// Tells us the variable bin to which a track would belong
unsigned int PurgeDuplicate::findVarRInvBin(const Tracklet* trk) const {
  std::vector<double> rInvBins = settings_.varRInvBins();

  //Get rinverse of track
  double rInv = trk->rinv();

  //Check between what 2 values in rinvbins rinv is between
  auto bins = std::upper_bound(rInvBins.begin(), rInvBins.end(), rInv);

  //return integer for bin index
  unsigned int rIndx = std::distance(rInvBins.begin(), bins);
  if (rIndx == std::distance(rInvBins.end(), bins))
    return rInvBins.size() - 2;
  else if (bins == rInvBins.begin())
    return std::distance(rInvBins.begin(), bins);
  else
    return rIndx - 1;
}

// Tells us the overlap bin(s) to which a track belongs
std::vector<unsigned int> PurgeDuplicate::findOverlapRInvBins(const Tracklet* trk) const {
  double rInv = trk->rinv();
  const double overlapSize = settings_.overlapSize();
  const std::vector<double>& varRInvBins = settings_.varRInvBins();
  std::vector<unsigned int> chosenBins;
  for (long unsigned int i = 0; i < varRInvBins.size() - 1; i++) {
    if ((rInv < varRInvBins[i + 1] + overlapSize) && (rInv > varRInvBins[i] - overlapSize)) {
      chosenBins.push_back(i);
    }
  }
  return chosenBins;
}

// Tells us if a track is in the current bin
bool PurgeDuplicate::isTrackInBin(const std::vector<unsigned int>& vec, unsigned int num) const {
  auto result = std::find(vec.begin(), vec.end(), num);
  bool found = (result != vec.end());
  return found;
}
