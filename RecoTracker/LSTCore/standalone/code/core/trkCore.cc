#include "trkCore.h"

//___________________________________________________________________________________________________________________________________________________________________________________________
bool goodEvent() {
  if (ana.specific_event_index >= 0) {
    if ((int)ana.looper.getCurrentEventIndex() != ana.specific_event_index)
      return false;
  }

  // If splitting jobs are requested then determine whether to process the event or not based on remainder
  if (ana.nsplit_jobs >= 0 and ana.job_index >= 0) {
    if (ana.looper.getNEventsProcessed() % ana.nsplit_jobs != (unsigned int)ana.job_index)
      return false;
  }

  if (ana.verbose >= 2)
    std::cout << " ana.looper.getCurrentEventIndex(): " << ana.looper.getCurrentEventIndex() << std::endl;

  return true;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runMiniDoublet(LSTEvent* event, int evt) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Mini-Doublet start " << evt << std::endl;
  my_timer.Start();
  event->createMiniDoublets();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float md_elapsed = my_timer.RealTime();

  if (ana.verbose >= 2)
    std::cout << evt << " Reco Mini-doublet processing time: " << md_elapsed << " secs" << std::endl;

  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced: " << event->getNumberOfMiniDoublets() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced barrel layer 1: " << event->getNumberOfMiniDoubletsByLayerBarrel(0)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced barrel layer 2: " << event->getNumberOfMiniDoubletsByLayerBarrel(1)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced barrel layer 3: " << event->getNumberOfMiniDoubletsByLayerBarrel(2)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced barrel layer 4: " << event->getNumberOfMiniDoubletsByLayerBarrel(3)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced barrel layer 5: " << event->getNumberOfMiniDoubletsByLayerBarrel(4)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced barrel layer 6: " << event->getNumberOfMiniDoubletsByLayerBarrel(5)
              << std::endl;

  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced endcap layer 1: " << event->getNumberOfMiniDoubletsByLayerEndcap(0)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced endcap layer 2: " << event->getNumberOfMiniDoubletsByLayerEndcap(1)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced endcap layer 3: " << event->getNumberOfMiniDoubletsByLayerEndcap(2)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced endcap layer 4: " << event->getNumberOfMiniDoubletsByLayerEndcap(3)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Mini-doublets produced endcap layer 5: " << event->getNumberOfMiniDoubletsByLayerEndcap(4)
              << std::endl;

  return md_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runSegment(LSTEvent* event) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Segment start" << std::endl;
  my_timer.Start();
  event->createSegmentsWithModuleMap();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float sg_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco Segment processing time: " << sg_elapsed << " secs" << std::endl;

  if (ana.verbose >= 2)
    std::cout << "# of Segments produced: " << event->getNumberOfSegments() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced layer 1-2: " << event->getNumberOfSegmentsByLayerBarrel(0) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced layer 2-3: " << event->getNumberOfSegmentsByLayerBarrel(1) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced layer 3-4: " << event->getNumberOfSegmentsByLayerBarrel(2) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced layer 4-5: " << event->getNumberOfSegmentsByLayerBarrel(3) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced layer 5-6: " << event->getNumberOfSegmentsByLayerBarrel(4) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced endcap layer 1: " << event->getNumberOfSegmentsByLayerEndcap(0) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced endcap layer 2: " << event->getNumberOfSegmentsByLayerEndcap(1) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced endcap layer 3: " << event->getNumberOfSegmentsByLayerEndcap(2) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced endcap layer 4: " << event->getNumberOfSegmentsByLayerEndcap(3) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Segments produced endcap layer 5: " << event->getNumberOfSegmentsByLayerEndcap(4) << std::endl;

  return sg_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runT3(LSTEvent* event) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco T3 start" << std::endl;
  my_timer.Start();
  event->createTriplets();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float t3_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco T3 processing time: " << t3_elapsed << " secs" << std::endl;

  if (ana.verbose >= 2)
    std::cout << "# of T3s produced: " << event->getNumberOfTriplets() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced layer 1-2-3: " << event->getNumberOfTripletsByLayerBarrel(0) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced layer 2-3-4: " << event->getNumberOfTripletsByLayerBarrel(1) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced layer 3-4-5: " << event->getNumberOfTripletsByLayerBarrel(2) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced layer 4-5-6: " << event->getNumberOfTripletsByLayerBarrel(3) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 1-2-3: " << event->getNumberOfTripletsByLayerEndcap(0) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 2-3-4: " << event->getNumberOfTripletsByLayerEndcap(1) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 3-4-5: " << event->getNumberOfTripletsByLayerEndcap(2) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 1: " << event->getNumberOfTripletsByLayerEndcap(0) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 2: " << event->getNumberOfTripletsByLayerEndcap(1) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 3: " << event->getNumberOfTripletsByLayerEndcap(2) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 4: " << event->getNumberOfTripletsByLayerEndcap(3) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T3s produced endcap layer 5: " << event->getNumberOfTripletsByLayerEndcap(4) << std::endl;

  return t3_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runpT3(LSTEvent* event) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Pixel Triplet pT3 start" << std::endl;
  my_timer.Start();
  event->createPixelTriplets();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float pt3_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco pT3 processing time: " << pt3_elapsed << " secs" << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Pixel T3s produced: " << event->getNumberOfPixelTriplets() << std::endl;

  return pt3_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runQuadruplet(LSTEvent* event) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Quadruplet start" << std::endl;
  my_timer.Start();
  event->createQuadruplets();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float t4_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco Quadruplet processing time: " << t4_elapsed << " secs" << std::endl;

  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced: " << event->getNumberOfQuadruplets() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced layer 1-2-3-4: " << event->getNumberOfQuadrupletsByLayerBarrel(0)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced layer 2: " << event->getNumberOfQuadrupletsByLayerBarrel(1) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced layer 3: " << event->getNumberOfQuadrupletsByLayerBarrel(2) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced layer 4: " << event->getNumberOfQuadrupletsByLayerBarrel(3) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced layer 5: " << event->getNumberOfQuadrupletsByLayerBarrel(4) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced layer 6: " << event->getNumberOfQuadrupletsByLayerBarrel(5) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced endcap layer 1: " << event->getNumberOfQuadrupletsByLayerEndcap(0)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced endcap layer 2: " << event->getNumberOfQuadrupletsByLayerEndcap(1)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced endcap layer 3: " << event->getNumberOfQuadrupletsByLayerEndcap(2)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced endcap layer 4: " << event->getNumberOfQuadrupletsByLayerEndcap(3)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quadruplets produced endcap layer 5: " << event->getNumberOfQuadrupletsByLayerEndcap(4)
              << std::endl;

  return t4_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runQuintuplet(LSTEvent* event) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Quintuplet start" << std::endl;
  my_timer.Start();
  event->createQuintuplets();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float t5_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco Quintuplet processing time: " << t5_elapsed << " secs" << std::endl;

  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced: " << event->getNumberOfQuintuplets() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced layer 1-2-3-4-5-6: " << event->getNumberOfQuintupletsByLayerBarrel(0)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced layer 2: " << event->getNumberOfQuintupletsByLayerBarrel(1) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced layer 3: " << event->getNumberOfQuintupletsByLayerBarrel(2) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced layer 4: " << event->getNumberOfQuintupletsByLayerBarrel(3) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced layer 5: " << event->getNumberOfQuintupletsByLayerBarrel(4) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced layer 6: " << event->getNumberOfQuintupletsByLayerBarrel(5) << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced endcap layer 1: " << event->getNumberOfQuintupletsByLayerEndcap(0)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced endcap layer 2: " << event->getNumberOfQuintupletsByLayerEndcap(1)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced endcap layer 3: " << event->getNumberOfQuintupletsByLayerEndcap(2)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced endcap layer 4: " << event->getNumberOfQuintupletsByLayerEndcap(3)
              << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Quintuplets produced endcap layer 5: " << event->getNumberOfQuintupletsByLayerEndcap(4)
              << std::endl;

  return t5_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runPixelLineSegment(LSTEvent* event, bool no_pls_dupclean) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Pixel Line Segment start" << std::endl;
  my_timer.Start();
  event->addPixelSegmentToEventFinalize();
  event->pixelLineSegmentCleaning(no_pls_dupclean);
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float pls_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco Pixel Line Segment processing time: " << pls_elapsed << " secs" << std::endl;

  return pls_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runPixelQuintuplet(LSTEvent* event) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Pixel Quintuplet start" << std::endl;
  my_timer.Start();
  event->createPixelQuintuplets();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float pt5_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco Pixel Quintuplet processing time: " << pt5_elapsed << " secs" << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Pixel Quintuplets produced: " << event->getNumberOfPixelQuintuplets() << std::endl;

  return pt5_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runTrackCandidate(LSTEvent* event, bool no_pls_dupclean, bool tc_pls_triplets) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco TrackCandidate start" << std::endl;
  my_timer.Start();
  event->createTrackCandidates(no_pls_dupclean, tc_pls_triplets);
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float tc_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco TrackCandidate processing time: " << tc_elapsed << " secs" << std::endl;

  if (ana.verbose >= 2)
    std::cout << "# of TrackCandidates produced: " << event->getNumberOfTrackCandidates() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of Pixel TrackCandidates produced: " << event->getNumberOfPixelTrackCandidates() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "    # of pT5 TrackCandidates produced: " << event->getNumberOfPT5TrackCandidates() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "    # of pT3 TrackCandidates produced: " << event->getNumberOfPT3TrackCandidates() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "    # of pLS TrackCandidates produced: " << event->getNumberOfPLSTrackCandidates() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T5 TrackCandidates produced: " << event->getNumberOfT5TrackCandidates() << std::endl;
  if (ana.verbose >= 2)
    std::cout << "# of T4 TrackCandidates produced: " << event->getNumberOfT4TrackCandidates() << std::endl;
  if (ana.verbose >= 2)
    printf("[MEM] Total: %.1f MB\n", event->getMemoryAllocatedMB());

  return tc_elapsed;
}

//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------

//___________________________________________________________________________________________________________________________________________________________________________________________
std::vector<int> matchedSimTrkIdxs(std::vector<unsigned int> hitidxs,
                                   std::vector<unsigned int> hittypes,
                                   std::vector<int> const& trk_simhit_simTrkIdx,
                                   std::vector<std::vector<int>> const& trk_ph2_simHitIdx,
                                   std::vector<std::vector<int>> const& trk_pix_simHitIdx,
                                   bool verbose,
                                   float matchfrac,
                                   float* pmatched) {
  std::vector<int> matched_idxs;
  std::vector<float> matched_idx_fracs;
  std::tie(matched_idxs, matched_idx_fracs) = matchedSimTrkIdxsAndFracs(
      hitidxs, hittypes, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx, verbose, matchfrac, pmatched);
  return matched_idxs;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
std::tuple<std::vector<int>, std::vector<float>> matchedSimTrkIdxsAndFracs(
    std::vector<unsigned int> hitidxs,
    std::vector<unsigned int> hittypes,
    std::vector<int> const& trk_simhit_simTrkIdx,
    std::vector<std::vector<int>> const& trk_ph2_simHitIdx,
    std::vector<std::vector<int>> const& trk_pix_simHitIdx,
    bool verbose,
    float matchfrac,
    float* pmatched) {
  if (hitidxs.size() != hittypes.size()) {
    std::cout << "Error: matched_sim_trk_idxs()   hitidxs and hittypes have different lengths" << std::endl;
    std::cout << "hitidxs.size(): " << hitidxs.size() << std::endl;
    std::cout << "hittypes.size(): " << hittypes.size() << std::endl;
  }

  std::vector<std::pair<unsigned int, unsigned int>> to_check_duplicate;
  for (size_t i = 0; i < hitidxs.size(); ++i) {
    auto hitidx = hitidxs[i];
    auto hittype = hittypes[i];
    auto item = std::make_pair(hitidx, hittype);
    if (std::find(to_check_duplicate.begin(), to_check_duplicate.end(), item) == to_check_duplicate.end()) {
      to_check_duplicate.push_back(item);
    }
  }

  int nhits_input = to_check_duplicate.size();

  std::vector<std::vector<int>> simtrk_idxs;
  std::vector<int> unique_idxs;  // to aggregate which ones to count and test

  if (verbose) {
    std::cout << " '------------------------': "
              << "------------------------" << std::endl;
  }

  for (size_t ihit = 0; ihit < to_check_duplicate.size(); ++ihit) {
    auto ihitdata = to_check_duplicate[ihit];
    auto&& [hitidx, hittype] = ihitdata;

    if (verbose) {
      std::cout << " hitidx: " << hitidx << " hittype: " << hittype << std::endl;
    }

    std::vector<int> simtrk_idxs_per_hit;

    const std::vector<std::vector<int>>* simHitIdxs = hittype == 4 ? &trk_ph2_simHitIdx : &trk_pix_simHitIdx;

    if (verbose) {
      std::cout << " trk_ph2_simHitIdx.size(): " << trk_ph2_simHitIdx.size() << std::endl;
      std::cout << " trk_pix_simHitIdx.size(): " << trk_pix_simHitIdx.size() << std::endl;
    }

    if (static_cast<const unsigned int>((*simHitIdxs).size()) <= hitidx) {
      std::cout << "ERROR" << std::endl;
      std::cout << " hittype: " << hittype << std::endl;
      std::cout << " trk_pix_simHitIdx.size(): " << trk_pix_simHitIdx.size() << std::endl;
      std::cout << " trk_ph2_simHitIdx.size(): " << trk_ph2_simHitIdx.size() << std::endl;
      std::cout << (*simHitIdxs).size() << " " << hittype << std::endl;
      std::cout << hitidx << " " << hittype << std::endl;
    }

    for (auto& simhit_idx : (*simHitIdxs).at(hitidx)) {
      if (static_cast<const int>(trk_simhit_simTrkIdx.size()) <= simhit_idx) {
        std::cout << (*simHitIdxs).size() << " " << hittype << std::endl;
        std::cout << hitidx << " " << hittype << std::endl;
        std::cout << trk_simhit_simTrkIdx.size() << " " << simhit_idx << std::endl;
      }
      int simtrk_idx = trk_simhit_simTrkIdx[simhit_idx];
      if (verbose) {
        std::cout << " hitidx: " << hitidx << " simhit_idx: " << simhit_idx << " simtrk_idx: " << simtrk_idx
                  << std::endl;
      }
      simtrk_idxs_per_hit.push_back(simtrk_idx);
      if (std::find(unique_idxs.begin(), unique_idxs.end(), simtrk_idx) == unique_idxs.end())
        unique_idxs.push_back(simtrk_idx);
    }

    if (simtrk_idxs_per_hit.size() == 0) {
      if (verbose) {
        std::cout << " hitidx: " << hitidx << " -1: " << -1 << std::endl;
      }
      simtrk_idxs_per_hit.push_back(-1);
      if (std::find(unique_idxs.begin(), unique_idxs.end(), -1) == unique_idxs.end())
        unique_idxs.push_back(-1);
    }

    simtrk_idxs.push_back(simtrk_idxs_per_hit);
  }

  if (verbose) {
    std::cout << " unique_idxs.size(): " << unique_idxs.size() << std::endl;
    for (auto& unique_idx : unique_idxs) {
      std::cout << " unique_idx: " << unique_idx << std::endl;
    }
  }

  // print
  if (verbose) {
    std::cout << "va print" << std::endl;
    for (auto& vec : simtrk_idxs) {
      for (auto& idx : vec) {
        std::cout << idx << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "va print end" << std::endl;
  }

  // Compute all permutations
  std::function<void(std::vector<std::vector<int>>&, std::vector<int>, size_t, std::vector<std::vector<int>>&)> perm =
      [&](std::vector<std::vector<int>>& result,
          std::vector<int> intermediate,
          size_t n,
          std::vector<std::vector<int>>& va) {
        if (va.size() > n) {
          for (auto x : va[n]) {
            std::vector<int> copy_intermediate(intermediate);
            copy_intermediate.push_back(x);
            perm(result, copy_intermediate, n + 1, va);
          }
        } else {
          result.push_back(intermediate);
        }
      };

  std::vector<std::vector<int>> allperms;
  perm(allperms, std::vector<int>(), 0, simtrk_idxs);

  if (verbose) {
    std::cout << " allperms.size(): " << allperms.size() << std::endl;
    for (unsigned iperm = 0; iperm < allperms.size(); ++iperm) {
      std::cout << " allperms[iperm].size(): " << allperms[iperm].size() << std::endl;
      for (unsigned ielem = 0; ielem < allperms[iperm].size(); ++ielem) {
        std::cout << " allperms[iperm][ielem]: " << allperms[iperm][ielem] << std::endl;
      }
    }
  }
  int maxHitMatchCount = 0;  // ultimate maximum of the number of matched hits
  std::vector<int> matched_sim_trk_idxs;
  std::vector<float> matched_sim_trk_idxs_frac;
  float max_percent_matched = 0.0f;
  for (auto& trkidx_perm : allperms) {
    std::vector<int> counts;
    for (auto& unique_idx : unique_idxs) {
      int cnt = std::count(trkidx_perm.begin(), trkidx_perm.end(), unique_idx);
      counts.push_back(cnt);
    }
    auto result = std::max_element(counts.begin(), counts.end());
    int rawidx = std::distance(counts.begin(), result);
    int trkidx = unique_idxs[rawidx];
    if (trkidx < 0)
      continue;
    float percent_matched = static_cast<float>(counts[rawidx]) / nhits_input;
    if (verbose) {
      std::cout << " fr: " << percent_matched << std::endl;
    }
    if (percent_matched > matchfrac) {
      matched_sim_trk_idxs.push_back(trkidx);
      matched_sim_trk_idxs_frac.push_back(percent_matched);
    }
    maxHitMatchCount = std::max(maxHitMatchCount, *std::max_element(counts.begin(), counts.end()));
    max_percent_matched = std::max(max_percent_matched, percent_matched);
  }

  // If pmatched is provided, set its value
  if (pmatched != nullptr) {
    *pmatched = max_percent_matched;
  }

  std::map<int, float> pairs;
  unsigned size = matched_sim_trk_idxs.size();
  for (unsigned i = 0; i < size; ++i) {
    int idx = matched_sim_trk_idxs[i];
    float frac = matched_sim_trk_idxs_frac[i];
    if (pairs.find(idx) != pairs.end()) {
      if (pairs[idx] < frac)
        pairs[idx] = frac;
    } else {
      pairs[idx] = frac;
    }
  }
  std::vector<int> result;
  std::vector<float> result_frac;
  // Loop over the map using range-based for loop
  for (const auto& pair : pairs) {
    result.push_back(pair.first);
    result_frac.push_back(pair.second);
  }
  // Sort indices based on 'values'
  auto indices = sort_indices(result_frac);
  // Reorder 'vec1' and 'vec2' based on the sorted indices
  std::vector<int> sorted_result(result.size());
  std::vector<float> sorted_result_frac(result_frac.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    sorted_result[i] = result[indices[i]];
    sorted_result_frac[i] = result_frac[indices[i]];
  }
  return std::make_tuple(sorted_result, sorted_result_frac);
}

//__________________________________________________________________________________________
int getDenomSimTrkType(int isimtrk,
                       std::vector<int> const& trk_sim_q,
                       std::vector<float> const& trk_sim_pt,
                       std::vector<float> const& trk_sim_eta,
                       std::vector<int> const& trk_sim_bunchCrossing,
                       std::vector<int> const& trk_sim_event,
                       std::vector<int> const& trk_sim_parentVtxIdx,
                       std::vector<float> const& trk_simvtx_x,
                       std::vector<float> const& trk_simvtx_y,
                       std::vector<float> const& trk_simvtx_z) {
  if (isimtrk < 0)
    return 0;  // not a sim
  const int& q = trk_sim_q[isimtrk];
  if (q == 0)
    return 1;  // sim
  const float& pt = trk_sim_pt[isimtrk];
  const float& eta = trk_sim_eta[isimtrk];
  if (pt < 1 or std::abs(eta) > 2.4)
    return 2;  // sim and charged
  const int& bunch = trk_sim_bunchCrossing[isimtrk];
  const int& event = trk_sim_event[isimtrk];
  const int& vtxIdx = trk_sim_parentVtxIdx[isimtrk];
  const float& vtx_x = trk_simvtx_x[isimtrk];
  const float& vtx_y = trk_simvtx_y[isimtrk];
  const float& vtx_z = trk_simvtx_z[isimtrk];
  const float& vtx_perp = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);
  if (vtx_perp > 2.5)
    return 3;  // pt > 1 and abs(eta) < 2.4
  if (std::abs(vtx_z) > 30)
    return 4;  // pt > 1 and abs(eta) < 2.4 and vtx < 2.5
  if (bunch != 0)
    return 5;  // pt > 1 and abs(eta) < 2.4 and vtx < 2.5 and vtx < 300
  if (event != 0)
    return 6;  // pt > 1 and abs(eta) < 2.4 and vtx 2.5/30 and bunch == 0
  return 7;    // pt > 1 and abs(eta) < 2.4 and vtx 2.5/30 and bunch == 0 and event == 0
}

//__________________________________________________________________________________________
int getDenomSimTrkType(std::vector<int> simidxs,
                       std::vector<int> const& trk_sim_q,
                       std::vector<float> const& trk_sim_pt,
                       std::vector<float> const& trk_sim_eta,
                       std::vector<int> const& trk_sim_bunchCrossing,
                       std::vector<int> const& trk_sim_event,
                       std::vector<int> const& trk_sim_parentVtxIdx,
                       std::vector<float> const& trk_simvtx_x,
                       std::vector<float> const& trk_simvtx_y,
                       std::vector<float> const& trk_simvtx_z) {
  int type = 0;
  for (auto& simidx : simidxs) {
    int this_type = getDenomSimTrkType(simidx,
                                       trk_sim_q,
                                       trk_sim_pt,
                                       trk_sim_eta,
                                       trk_sim_bunchCrossing,
                                       trk_sim_event,
                                       trk_sim_parentVtxIdx,
                                       trk_simvtx_x,
                                       trk_simvtx_y,
                                       trk_simvtx_z);
    if (this_type > type) {
      type = this_type;
    }
  }
  return type;
}

//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------

//___________________________________________________________________________________________________________________________________________________________________________________________
float drfracSimHitConsistentWithHelix(int isimtrk,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simvtx_x,
                                      std::vector<float> const& trk_simvtx_y,
                                      std::vector<float> const& trk_simvtx_z,
                                      std::vector<float> const& trk_sim_pt,
                                      std::vector<float> const& trk_sim_eta,
                                      std::vector<float> const& trk_sim_phi,
                                      std::vector<int> const& trk_sim_q,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z) {
  // Read track parameters
  float vx = trk_simvtx_x[0];
  float vy = trk_simvtx_y[0];
  float vz = trk_simvtx_z[0];
  float pt = trk_sim_pt[isimtrk];
  float eta = trk_sim_eta[isimtrk];
  float phi = trk_sim_phi[isimtrk];
  int charge = trk_sim_q[isimtrk];

  // Construct helix object
  lst_math::Helix helix(pt, eta, phi, vx, vy, vz, charge);

  return drfracSimHitConsistentWithHelix(helix, isimhitidx, trk_simhit_x, trk_simhit_y, trk_simhit_z);
}

//__________________________________________________________________________________________
float drfracSimHitConsistentWithHelix(lst_math::Helix& helix,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z) {
  // Sim hit vector
  std::vector<float> point = {trk_simhit_x[isimhitidx], trk_simhit_y[isimhitidx], trk_simhit_z[isimhitidx]};

  // Inferring parameter t of helix parametric function via z position
  float t = helix.infer_t(point);

  // If the best fit is more than pi parameter away then it's a returning hit and should be ignored
  if (not(t <= M_PI))
    return 999;

  // Expected hit position with given z
  auto [x, y, z, r] = helix.get_helix_point(t);

  // ( expected_r - simhit_r ) / expected_r
  float drfrac = std::abs(helix.compare_radius(point)) / r;

  return drfrac;
}

//__________________________________________________________________________________________
float distxySimHitConsistentWithHelix(int isimtrk,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simvtx_x,
                                      std::vector<float> const& trk_simvtx_y,
                                      std::vector<float> const& trk_simvtx_z,
                                      std::vector<float> const& trk_sim_pt,
                                      std::vector<float> const& trk_sim_eta,
                                      std::vector<float> const& trk_sim_phi,
                                      std::vector<int> const& trk_sim_q,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z) {
  // Read track parameters
  float vx = trk_simvtx_x[0];
  float vy = trk_simvtx_y[0];
  float vz = trk_simvtx_z[0];
  float pt = trk_sim_pt[isimtrk];
  float eta = trk_sim_eta[isimtrk];
  float phi = trk_sim_phi[isimtrk];
  int charge = trk_sim_q[isimtrk];

  // Construct helix object
  lst_math::Helix helix(pt, eta, phi, vx, vy, vz, charge);

  return distxySimHitConsistentWithHelix(helix, isimhitidx, trk_simhit_x, trk_simhit_y, trk_simhit_z);
}

//__________________________________________________________________________________________
float distxySimHitConsistentWithHelix(lst_math::Helix& helix,
                                      int isimhitidx,
                                      std::vector<float> const& trk_simhit_x,
                                      std::vector<float> const& trk_simhit_y,
                                      std::vector<float> const& trk_simhit_z) {
  // Sim hit vector
  std::vector<float> point = {trk_simhit_x[isimhitidx], trk_simhit_y[isimhitidx], trk_simhit_z[isimhitidx]};

  // Inferring parameter t of helix parametric function via z position
  float t = helix.infer_t(point);

  // If the best fit is more than pi parameter away then it's a returning hit and should be ignored
  if (not(t <= M_PI))
    return 999;

  // Expected hit position with given z
  //auto [x, y, z, r] = helix.get_helix_point(t);

  // ( expected_r - simhit_r ) / expected_r
  float distxy = helix.compare_xy(point);

  return distxy;
}

//__________________________________________________________________________________________
TVector3 calculateR3FromPCA(const TVector3& p3, const float dxy, const float dz) {
  const float pt = p3.Pt();
  const float p = p3.Mag();
  const float vz = dz * pt * pt / p / p;

  const float vx = -dxy * p3.y() / pt - p3.x() / p * p3.z() / p * dz;
  const float vy = dxy * p3.x() / pt - p3.y() / p * p3.z() / p * dz;
  return TVector3(vx, vy, vz);
}

//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------

//___________________________________________________________________________________________________________________________________________________________________________________________
float addInputsToEventPreLoad(LSTEvent* event,
                              lst::LSTInputHostCollection* lstInputHC,
                              LSTInputDeviceCollection* lstInputDC,
                              ALPAKA_ACCELERATOR_NAMESPACE::Queue& queue) {
  TStopwatch my_timer;

  if (ana.verbose >= 2)
    std::cout << "Loading Inputs (i.e. outer tracker hits, and pixel line segements) to the Line Segment Tracking.... "
              << std::endl;

  my_timer.Start();

  // We can't use CopyToDevice because the device can be DevHost
  alpaka::memcpy(queue, lstInputDC->buffer(), lstInputHC->buffer());
  alpaka::wait(queue);

  event->addInputToEvent(lstInputDC);
  event->addHitToEvent();

  event->addPixelSegmentToEventStart();
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float hit_loading_elapsed = my_timer.RealTime();

  if (ana.verbose >= 2)
    std::cout << "Loading inputs processing time: " << hit_loading_elapsed << " secs" << std::endl;

  return hit_loading_elapsed;
}

//________________________________________________________________________________________________________________________________
void printTimingInformation(std::vector<std::vector<float>>& timing_information, float fullTime, float fullavg) {
  if (ana.verbose == 0)
    return;

  std::cout << std::showpoint;
  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  std::cout << std::right;
  std::cout << "Timing summary" << std::endl;
  std::cout << std::setw(6) << "Evt";
  std::cout << "   " << std::setw(6) << "Hits";
  std::cout << "   " << std::setw(6) << "MD";
  std::cout << "   " << std::setw(6) << "LS";
  std::cout << "   " << std::setw(6) << "T3";
  std::cout << "   " << std::setw(6) << "T5";
  std::cout << "   " << std::setw(6) << "pLS";
  std::cout << "   " << std::setw(6) << "T4";
  std::cout << "   " << std::setw(6) << "pT5";
  std::cout << "   " << std::setw(6) << "pT3";
  std::cout << "   " << std::setw(6) << "TC";
  std::cout << "   " << std::setw(6) << "Reset";
  std::cout << "   " << std::setw(7) << "Total";
  std::cout << "   " << std::setw(7) << "Total(short)";
  std::cout << std::endl;
  std::vector<float> timing_sum_information(timing_information[0].size());
  std::vector<float> timing_shortlist;
  std::vector<float> timing_list;
  for (size_t ievt = 0; ievt < timing_information.size(); ++ievt) {
    auto timing = timing_information[ievt];
    float timing_total = 0.f;
    float timing_total_short = 0.f;
    timing_total += timing[0] * 1000;           // Hits
    for (size_t iobj = 1; iobj <= 9; ++iobj) {  // MD-TC
      timing_total += timing[iobj] * 1000;
      if (iobj != 5)
        timing_total_short += timing[iobj] * 1000;  // exclude pLS
    }
    timing_total_short += timing[10] * 1000;  // Reset
    std::cout << std::setw(6) << ievt;
    for (auto objtime : timing) {
      std::cout << "   " << std::setw(6) << objtime * 1000;  // Print Hits-Reset
    }
    std::cout << "   " << std::setw(7) << timing_total;        // Total time
    std::cout << "   " << std::setw(7) << timing_total_short;  // Total time
    std::cout << std::endl;
    for (size_t iobj = 0; iobj <= 10; ++iobj) {  // Hits-Reset
      timing_sum_information[iobj] += timing[iobj] * 1000;
    }
    timing_shortlist.push_back(timing_total_short);  // short total
    timing_list.push_back(timing_total);             // short total
  }
  for (size_t iobj = 0; iobj <= 10; iobj++) {  // Hits-Reset
    timing_sum_information[iobj] /= timing_information.size();
  }

  float timing_total_avg = 0.0;
  float timing_totalshort_avg = 0.0;
  timing_total_avg += timing_sum_information[0];  // Hits
  for (size_t iobj = 1; iobj <= 10; iobj++) {     // MD-Reset
    timing_total_avg += timing_sum_information[iobj];
    if (iobj != 5)
      timing_totalshort_avg += timing_sum_information[iobj];  // exclude pLS
  }

  float standardDeviation = 0.0;
  for (auto shorttime : timing_shortlist) {
    standardDeviation += pow(shorttime - timing_totalshort_avg, 2);
  }
  float stdDev = sqrt(standardDeviation / timing_shortlist.size());

  std::cout << std::setprecision(1);
  std::cout << std::setw(6) << "Evt";
  std::cout << "   " << std::setw(6) << "Hits";
  std::cout << "   " << std::setw(6) << "MD";
  std::cout << "   " << std::setw(6) << "LS";
  std::cout << "   " << std::setw(6) << "T3";
  std::cout << "   " << std::setw(6) << "T5";
  std::cout << "   " << std::setw(6) << "pLS";
  std::cout << "   " << std::setw(6) << "T4";
  std::cout << "   " << std::setw(6) << "pT5";
  std::cout << "   " << std::setw(6) << "pT3";
  std::cout << "   " << std::setw(6) << "TC";
  std::cout << "   " << std::setw(6) << "Reset";
  std::cout << "   " << std::setw(7) << "Total";
  std::cout << "   " << std::setw(7) << "Total(short)";
  std::cout << std::endl;
  std::cout << std::setw(6) << "avg";
  for (auto objsum : timing_sum_information) {
    std::cout << "   " << std::setw(6) << objsum;  // Print Hits-Reset
  }
  std::cout << "   " << std::setw(7) << timing_total_avg;       // Average total time
  std::cout << "   " << std::setw(7) << timing_totalshort_avg;  // Average total time
  std::cout << "+/- " << std::setw(4) << stdDev;
  std::cout << "   " << std::setw(7) << fullavg;  // Average full time
  std::cout << "   " << ana.compilation_target;
  std::cout << "[s=" << ana.streams << "]";
  std::cout << std::endl;

  std::cout << std::left;
}

//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------

//__________________________________________________________________________________________
TString get_absolute_path_after_check_file_exists(const std::string name) {
  std::filesystem::path fullpath = std::filesystem::absolute(name.c_str());
  // std::cout << "Checking file path = " << fullpath << std::endl;
  // std::cout <<  " fullpath.string().c_str(): " << fullpath.string().c_str() <<  std::endl;
  if (not std::filesystem::exists(fullpath)) {
    std::cout << "ERROR: Could not find the file = " << fullpath << std::endl;
    exit(2);
  }
  return TString(fullpath.string().c_str());
}

//_______________________________________________________________________________
void writeMetaData() {
  // Write out metadata of the code to the output_tfile
  ana.output_tfile->cd();
  gSystem->Exec(TString::Format("(cd $TRACKLOOPERDIR && echo '' && (cd - > /dev/null) ) > %s.gitversion.txt ",
                                ana.output_tfile->GetName()));
  gSystem->Exec(TString::Format("(cd $TRACKLOOPERDIR && git rev-parse HEAD && (cd - > /dev/null)) >> %s.gitversion.txt",
                                ana.output_tfile->GetName()));
  gSystem->Exec(TString::Format("(cd $TRACKLOOPERDIR && echo 'git status' && (cd - > /dev/null)) >> %s.gitversion.txt",
                                ana.output_tfile->GetName()));
  gSystem->Exec(
      TString::Format("(cd $TRACKLOOPERDIR && git  --no-pager status && (cd - > /dev/null)) >> %s.gitversion.txt",
                      ana.output_tfile->GetName()));
  gSystem->Exec(TString::Format(
      "(cd $TRACKLOOPERDIR && echo 'git --no-pager log -n 100' && (cd - > /dev/null)) >> %s.gitversion.txt",
      ana.output_tfile->GetName()));
  gSystem->Exec(TString::Format("(cd $TRACKLOOPERDIR && echo 'git diff' && (cd - > /dev/null)) >> %s.gitversion.txt",
                                ana.output_tfile->GetName()));
  gSystem->Exec(
      TString::Format("(cd $TRACKLOOPERDIR && git --no-pager diff  && (cd - > /dev/null)) >> %s.gitversion.txt",
                      ana.output_tfile->GetName()));

  // Write gitversion info
  std::ifstream t(TString::Format("%s.gitversion.txt", ana.output_tfile->GetName()));
  std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  TNamed code_tag_data("code_tag_data", str.c_str());
  ana.output_tfile->cd();
  code_tag_data.Write();
  gSystem->Exec(TString::Format("rm %s.gitversion.txt", ana.output_tfile->GetName()));

  // Write make script log
  TString make_log_path = TString::Format("%s/.make.log", ana.track_looper_dir_path.Data());
  std::ifstream makelog(make_log_path.Data());
  std::string makestr((std::istreambuf_iterator<char>(makelog)), std::istreambuf_iterator<char>());
  TNamed make_log("make_log", makestr.c_str());
  make_log.Write();

  // Write git diff output in a separate string to gauge the difference
  gSystem->Exec(TString::Format("(cd $TRACKLOOPERDIR && git --no-pager diff  && (cd - > /dev/null)) > %s.gitdiff.txt",
                                ana.output_tfile->GetName()));
  std::ifstream gitdiff(TString::Format("%s.gitdiff.txt", ana.output_tfile->GetName()));
  std::string strgitdiff((std::istreambuf_iterator<char>(gitdiff)), std::istreambuf_iterator<char>());
  TNamed gitdifftnamed("gitdiff", strgitdiff.c_str());
  gitdifftnamed.Write();
  gSystem->Exec(TString::Format("rm %s.gitdiff.txt", ana.output_tfile->GetName()));

  // Write Parse from makestr the TARGET
  TString maketstr = makestr.c_str();
  TString rawstrdata = maketstr.ReplaceAll("MAKETARGET=", "%");
  TString targetrawdata = RooUtil::StringUtil::rsplit(rawstrdata, "%")[1];
  TString targetdata = RooUtil::StringUtil::split(targetrawdata)[0];
  ana.compilation_target = targetdata.Data();

  // Write out input sample or file name
  TNamed input("input", ana.input_raw_string.Data());
  input.Write();

  // Write the full command line used
  TNamed full_cmd_line("full_cmd_line", ana.full_cmd_line.Data());
  full_cmd_line.Write();

  // Write the TRACKLOOPERDIR
  TNamed tracklooper_path("tracklooper_path", ana.track_looper_dir_path.Data());
  tracklooper_path.Write();
}
