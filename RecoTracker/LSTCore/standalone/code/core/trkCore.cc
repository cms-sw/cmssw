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
float runMiniDoublet(LSTEvent *event, int evt) {
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
float runSegment(LSTEvent *event) {
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
float runT3(LSTEvent *event) {
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
float runpT3(LSTEvent *event) {
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
float runQuintuplet(LSTEvent *event) {
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
float runPixelLineSegment(LSTEvent *event,
                          std::vector<unsigned int> hitIndices_vec0,
                          std::vector<unsigned int> hitIndices_vec1,
                          std::vector<unsigned int> hitIndices_vec2,
                          std::vector<unsigned int> hitIndices_vec3,
                          std::vector<float> deltaPhi_vec,
                          bool no_pls_dupclean) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco Pixel Line Segment start" << std::endl;
  my_timer.Start();
  event->addPixelSegmentToEventFinalize(
      hitIndices_vec0, hitIndices_vec1, hitIndices_vec2, hitIndices_vec3, deltaPhi_vec);
  event->pixelLineSegmentCleaning(no_pls_dupclean);
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float pls_elapsed = my_timer.RealTime();
  if (ana.verbose >= 2)
    std::cout << "Reco Pixel Line Segment processing time: " << pls_elapsed << " secs" << std::endl;

  return pls_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runPixelQuintuplet(LSTEvent *event) {
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
float runTrackCandidate(LSTEvent *event, bool no_pls_dupclean, bool tc_pls_triplets) {
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

  return tc_elapsed;
}

//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------
//  ---------------------------------- =========================================== ----------------------------------------------

//___________________________________________________________________________________________________________________________________________________________________________________________
std::vector<int> matchedSimTrkIdxs(std::vector<int> hitidxs, std::vector<int> hittypes, bool verbose) {
  std::vector<unsigned int> hitidxs_(std::begin(hitidxs), std::end(hitidxs));
  std::vector<unsigned int> hittypes_(std::begin(hittypes), std::end(hittypes));
  return matchedSimTrkIdxs(hitidxs_, hittypes_, verbose);
}

//___________________________________________________________________________________________________________________________________________________________________________________________
std::vector<int> matchedSimTrkIdxs(std::vector<unsigned int> hitidxs,
                                   std::vector<unsigned int> hittypes,
                                   bool verbose,
                                   float *pmatched) {
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
    auto &&[hitidx, hittype] = ihitdata;

    if (verbose) {
      std::cout << " hitidx: " << hitidx << " hittype: " << hittype << std::endl;
    }

    std::vector<int> simtrk_idxs_per_hit;

    const std::vector<std::vector<int>> *simHitIdxs = hittype == 4 ? &trk.ph2_simHitIdx() : &trk.pix_simHitIdx();

    if (verbose) {
      std::cout << " trk.ph2_simHitIdx().size(): " << trk.ph2_simHitIdx().size() << std::endl;
      std::cout << " trk.pix_simHitIdx().size(): " << trk.pix_simHitIdx().size() << std::endl;
    }

    if (static_cast<const unsigned int>((*simHitIdxs).size()) <= hitidx) {
      std::cout << "ERROR" << std::endl;
      std::cout << " hittype: " << hittype << std::endl;
      std::cout << " trk.pix_simHitIdx().size(): " << trk.pix_simHitIdx().size() << std::endl;
      std::cout << " trk.ph2_simHitIdx().size(): " << trk.ph2_simHitIdx().size() << std::endl;
      std::cout << (*simHitIdxs).size() << " " << hittype << std::endl;
      std::cout << hitidx << " " << hittype << std::endl;
    }

    for (auto &simhit_idx : (*simHitIdxs).at(hitidx)) {
      if (static_cast<const int>(trk.simhit_simTrkIdx().size()) <= simhit_idx) {
        std::cout << (*simHitIdxs).size() << " " << hittype << std::endl;
        std::cout << hitidx << " " << hittype << std::endl;
        std::cout << trk.simhit_simTrkIdx().size() << " " << simhit_idx << std::endl;
      }
      int simtrk_idx = trk.simhit_simTrkIdx().at(simhit_idx);
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
    for (auto &unique_idx : unique_idxs) {
      std::cout << " unique_idx: " << unique_idx << std::endl;
    }
  }

  // print
  if (verbose) {
    std::cout << "va print" << std::endl;
    for (auto &vec : simtrk_idxs) {
      for (auto &idx : vec) {
        std::cout << idx << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "va print end" << std::endl;
  }

  // Compute all permutations
  std::function<void(std::vector<std::vector<int>> &, std::vector<int>, size_t, std::vector<std::vector<int>> &)> perm =
      [&](std::vector<std::vector<int>> &result,
          std::vector<int> intermediate,
          size_t n,
          std::vector<std::vector<int>> &va) {
        // std::cout <<  " 'called': " << "called" <<  std::endl;
        if (va.size() > n) {
          for (auto x : va[n]) {
            // std::cout <<  " n: " << n <<  std::endl;
            // std::cout <<  " intermediate.size(): " << intermediate.size() <<  std::endl;
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
  float max_percent_matched = 0.0f;
  for (auto &trkidx_perm : allperms) {
    std::vector<int> counts;
    for (auto &unique_idx : unique_idxs) {
      int cnt = std::count(trkidx_perm.begin(), trkidx_perm.end(), unique_idx);
      counts.push_back(cnt);
    }
    auto result = std::max_element(counts.begin(), counts.end());
    int rawidx = std::distance(counts.begin(), result);
    int trkidx = unique_idxs[rawidx];
    if (trkidx < 0)
      continue;
    float percent_matched = static_cast<float>(counts[rawidx]) / nhits_input;
    if (percent_matched > 0.75f)
      matched_sim_trk_idxs.push_back(trkidx);
    maxHitMatchCount = std::max(maxHitMatchCount, *std::max_element(counts.begin(), counts.end()));
    max_percent_matched = std::max(max_percent_matched, percent_matched);
  }

  // If pmatched is provided, set its value
  if (pmatched != nullptr) {
    *pmatched = max_percent_matched;
  }

  std::set<int> s;
  unsigned size = matched_sim_trk_idxs.size();
  for (unsigned i = 0; i < size; ++i)
    s.insert(matched_sim_trk_idxs[i]);
  matched_sim_trk_idxs.assign(s.begin(), s.end());
  return matched_sim_trk_idxs;
}

//__________________________________________________________________________________________
int getDenomSimTrkType(int isimtrk) {
  if (isimtrk < 0)
    return 0;  // not a sim
  const int &q = trk.sim_q()[isimtrk];
  if (q == 0)
    return 1;  // sim
  const float &pt = trk.sim_pt()[isimtrk];
  const float &eta = trk.sim_eta()[isimtrk];
  if (pt < 1 or abs(eta) > 2.4)
    return 2;  // sim and charged
  const int &bunch = trk.sim_bunchCrossing()[isimtrk];
  const int &event = trk.sim_event()[isimtrk];
  const int &vtxIdx = trk.sim_parentVtxIdx()[isimtrk];
  const float &vtx_x = trk.simvtx_x()[vtxIdx];
  const float &vtx_y = trk.simvtx_y()[vtxIdx];
  const float &vtx_z = trk.simvtx_z()[vtxIdx];
  const float &vtx_perp = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);
  if (vtx_perp > 2.5)
    return 3;  // pt > 1 and abs(eta) < 2.4
  if (abs(vtx_z) > 30)
    return 4;  // pt > 1 and abs(eta) < 2.4 and vtx < 2.5
  if (bunch != 0)
    return 5;  // pt > 1 and abs(eta) < 2.4 and vtx < 2.5 and vtx < 300
  if (event != 0)
    return 6;  // pt > 1 and abs(eta) < 2.4 and vtx 2.5/30 and bunch == 0
  return 7;    // pt > 1 and abs(eta) < 2.4 and vtx 2.5/30 and bunch == 0 and event == 0
}

//__________________________________________________________________________________________
int getDenomSimTrkType(std::vector<int> simidxs) {
  int type = 0;
  for (auto &simidx : simidxs) {
    int this_type = getDenomSimTrkType(simidx);
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
float drfracSimHitConsistentWithHelix(int isimtrk, int isimhitidx) {
  // Read track parameters
  float vx = trk.simvtx_x()[0];
  float vy = trk.simvtx_y()[0];
  float vz = trk.simvtx_z()[0];
  float pt = trk.sim_pt()[isimtrk];
  float eta = trk.sim_eta()[isimtrk];
  float phi = trk.sim_phi()[isimtrk];
  int charge = trk.sim_q()[isimtrk];

  // Construct helix object
  lst_math::Helix helix(pt, eta, phi, vx, vy, vz, charge);

  return drfracSimHitConsistentWithHelix(helix, isimhitidx);
}

//__________________________________________________________________________________________
float drfracSimHitConsistentWithHelix(lst_math::Helix &helix, int isimhitidx) {
  // Sim hit vector
  std::vector<float> point = {trk.simhit_x()[isimhitidx], trk.simhit_y()[isimhitidx], trk.simhit_z()[isimhitidx]};

  // Inferring parameter t of helix parametric function via z position
  float t = helix.infer_t(point);

  // If the best fit is more than pi parameter away then it's a returning hit and should be ignored
  if (not(t <= M_PI))
    return 999;

  // Expected hit position with given z
  auto [x, y, z, r] = helix.get_helix_point(t);

  // ( expected_r - simhit_r ) / expected_r
  float drfrac = abs(helix.compare_radius(point)) / r;

  return drfrac;
}

//__________________________________________________________________________________________
float distxySimHitConsistentWithHelix(int isimtrk, int isimhitidx) {
  // Read track parameters
  float vx = trk.simvtx_x()[0];
  float vy = trk.simvtx_y()[0];
  float vz = trk.simvtx_z()[0];
  float pt = trk.sim_pt()[isimtrk];
  float eta = trk.sim_eta()[isimtrk];
  float phi = trk.sim_phi()[isimtrk];
  int charge = trk.sim_q()[isimtrk];

  // Construct helix object
  lst_math::Helix helix(pt, eta, phi, vx, vy, vz, charge);

  return distxySimHitConsistentWithHelix(helix, isimhitidx);
}

//__________________________________________________________________________________________
float distxySimHitConsistentWithHelix(lst_math::Helix &helix, int isimhitidx) {
  // Sim hit vector
  std::vector<float> point = {trk.simhit_x()[isimhitidx], trk.simhit_y()[isimhitidx], trk.simhit_z()[isimhitidx]};

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
TVector3 calculateR3FromPCA(const TVector3 &p3, const float dxy, const float dz) {
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
void addInputsToLineSegmentTrackingPreLoad(std::vector<std::vector<float>> &out_trkX,
                                           std::vector<std::vector<float>> &out_trkY,
                                           std::vector<std::vector<float>> &out_trkZ,
                                           std::vector<std::vector<unsigned int>> &out_hitId,
                                           std::vector<std::vector<unsigned int>> &out_hitIdxs,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec0,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec1,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec2,
                                           std::vector<std::vector<unsigned int>> &out_hitIndices_vec3,
                                           std::vector<std::vector<float>> &out_deltaPhi_vec,
                                           std::vector<std::vector<float>> &out_ptIn_vec,
                                           std::vector<std::vector<float>> &out_ptErr_vec,
                                           std::vector<std::vector<float>> &out_px_vec,
                                           std::vector<std::vector<float>> &out_py_vec,
                                           std::vector<std::vector<float>> &out_pz_vec,
                                           std::vector<std::vector<float>> &out_eta_vec,
                                           std::vector<std::vector<float>> &out_etaErr_vec,
                                           std::vector<std::vector<float>> &out_phi_vec,
                                           std::vector<std::vector<int>> &out_charge_vec,
                                           std::vector<std::vector<unsigned int>> &out_seedIdx_vec,
                                           std::vector<std::vector<int>> &out_superbin_vec,
                                           std::vector<std::vector<PixelType>> &out_pixelType_vec,
                                           std::vector<std::vector<char>> &out_isQuad_vec) {
  unsigned int count = 0;
  auto n_see = trk.see_stateTrajGlbPx().size();
  std::vector<float> px_vec;
  px_vec.reserve(n_see);
  std::vector<float> py_vec;
  py_vec.reserve(n_see);
  std::vector<float> pz_vec;
  pz_vec.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec0;
  hitIndices_vec0.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec1;
  hitIndices_vec1.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec2;
  hitIndices_vec2.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec3;
  hitIndices_vec3.reserve(n_see);
  std::vector<float> ptIn_vec;
  ptIn_vec.reserve(n_see);
  std::vector<float> ptErr_vec;
  ptErr_vec.reserve(n_see);
  std::vector<float> etaErr_vec;
  etaErr_vec.reserve(n_see);
  std::vector<float> eta_vec;
  eta_vec.reserve(n_see);
  std::vector<float> phi_vec;
  phi_vec.reserve(n_see);
  std::vector<int> charge_vec;
  charge_vec.reserve(n_see);
  std::vector<unsigned int> seedIdx_vec;
  seedIdx_vec.reserve(n_see);
  std::vector<float> deltaPhi_vec;
  deltaPhi_vec.reserve(n_see);
  std::vector<float> trkX = trk.ph2_x();
  std::vector<float> trkY = trk.ph2_y();
  std::vector<float> trkZ = trk.ph2_z();
  std::vector<unsigned int> hitId = trk.ph2_detId();
  std::vector<unsigned int> hitIdxs(trk.ph2_detId().size());

  std::vector<int> superbin_vec;
  std::vector<PixelType> pixelType_vec;
  std::vector<char> isQuad_vec;
  std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
  const int hit_size = trkX.size();

  for (size_t iSeed = 0; iSeed < trk.see_stateTrajGlbPx().size(); ++iSeed) {
    //// track algorithm; partial copy from TrackBase.h
    // enum class TrackAlgorithm {
    //    undefAlgorithm = 0,
    //    ctf = 1,
    //    duplicateMerge = 2,
    //    cosmics = 3,
    //    initialStep = 4,
    //    lowPtTripletStep = 5,
    //    pixelPairStep = 6,
    //    detachedTripletStep = 7,
    //    mixedTripletStep = 8,
    //    pixelLessStep = 9,
    //    tobTecStep = 10,
    //    jetCoreRegionalStep = 11,
    //    conversionStep = 12,
    //    muonSeededStepInOut = 13,
    //    muonSeededStepOutIn = 14,
    //    outInEcalSeededConv = 15, inOutEcalSeededConv = 16,
    //    nuclInter = 17,
    //    standAloneMuon = 18, globalMuon = 19, cosmicStandAloneMuon = 20, cosmicGlobalMuon = 21,
    //    // Phase1
    //    highPtTripletStep = 22, lowPtQuadStep = 23, detachedQuadStep = 24,
    //    reservedForUpgrades1 = 25, reservedForUpgrades2 = 26,
    //    bTagGhostTracks = 27,
    //    beamhalo = 28,
    //    gsf = 29
    //};
    bool good_seed_type = false;
    if (trk.see_algo()[iSeed] == 4)
      good_seed_type = true;
    // if (trk.see_algo()[iSeed] == 5) good_seed_type = true;
    // if (trk.see_algo()[iSeed] == 7) good_seed_type = true;
    if (trk.see_algo()[iSeed] == 22)
      good_seed_type = true;
    // if (trk.see_algo()[iSeed] == 23) good_seed_type = true;
    // if (trk.see_algo()[iSeed] == 24) good_seed_type = true;
    if (not good_seed_type)
      continue;

    TVector3 p3LH(trk.see_stateTrajGlbPx()[iSeed], trk.see_stateTrajGlbPy()[iSeed], trk.see_stateTrajGlbPz()[iSeed]);
    float ptIn = p3LH.Pt();
    float eta = p3LH.Eta();
    float ptErr = trk.see_ptErr()[iSeed];

    if ((ptIn > ana.ptCut - 2 * ptErr)) {
      TVector3 r3LH(trk.see_stateTrajGlbX()[iSeed], trk.see_stateTrajGlbY()[iSeed], trk.see_stateTrajGlbZ()[iSeed]);
      TVector3 p3PCA(trk.see_px()[iSeed], trk.see_py()[iSeed], trk.see_pz()[iSeed]);
      TVector3 r3PCA(calculateR3FromPCA(p3PCA, trk.see_dxy()[iSeed], trk.see_dz()[iSeed]));
      TVector3 seedSD_mdRef_r3 = r3PCA;
      TVector3 seedSD_mdOut_r3 = r3LH;
      TVector3 seedSD_r3 = r3LH;
      TVector3 seedSD_p3 = p3LH;

      // The charge could be used directly in the line below
      float pixelSegmentDeltaPhiChange = r3LH.DeltaPhi(p3LH);
      float etaErr = trk.see_etaErr()[iSeed];
      float px = p3LH.X();
      float py = p3LH.Y();
      float pz = p3LH.Z();
      int charge = trk.see_q()[iSeed];
      unsigned int seedIdx = iSeed;

      PixelType pixtype = PixelType::kInvalid;
      if (ptIn >= 2.0) {
        pixtype = PixelType::kHighPt;
      } else if (ptIn >= (ana.ptCut - 2 * ptErr) and ptIn < 2.0) {
        if (pixelSegmentDeltaPhiChange >= 0) {
          pixtype = PixelType::kLowPtPosCurv;
        } else {
          pixtype = PixelType::kLowPtNegCurv;
        }
      } else {
        continue;
      }

      unsigned int hitIdx0 = hit_size + count;
      count++;

      unsigned int hitIdx1 = hit_size + count;
      count++;

      unsigned int hitIdx2 = hit_size + count;
      count++;

      unsigned int hitIdx3;
      if (trk.see_hitIdx()[iSeed].size() <= 3) {
        hitIdx3 = hitIdx2;
      } else {
        hitIdx3 = hit_size + count;
        count++;
      }

      trkX.push_back(r3PCA.X());
      trkY.push_back(r3PCA.Y());
      trkZ.push_back(r3PCA.Z());
      trkX.push_back(p3PCA.Pt());
      float p3PCA_Eta = p3PCA.Eta();
      trkY.push_back(p3PCA_Eta);
      float p3PCA_Phi = p3PCA.Phi();
      trkZ.push_back(p3PCA_Phi);
      trkX.push_back(r3LH.X());
      trkY.push_back(r3LH.Y());
      trkZ.push_back(r3LH.Z());
      hitId.push_back(1);
      hitId.push_back(1);
      hitId.push_back(1);
      if (trk.see_hitIdx()[iSeed].size() > 3) {
        trkX.push_back(r3LH.X());
        trkY.push_back(trk.see_dxy()[iSeed]);
        trkZ.push_back(trk.see_dz()[iSeed]);
        hitId.push_back(1);
      }
      px_vec.push_back(px);
      py_vec.push_back(py);
      pz_vec.push_back(pz);

      hitIndices_vec0.push_back(hitIdx0);
      hitIndices_vec1.push_back(hitIdx1);
      hitIndices_vec2.push_back(hitIdx2);
      hitIndices_vec3.push_back(hitIdx3);
      ptIn_vec.push_back(ptIn);
      ptErr_vec.push_back(ptErr);
      etaErr_vec.push_back(etaErr);
      eta_vec.push_back(eta);
      float phi = p3LH.Phi();
      phi_vec.push_back(phi);
      charge_vec.push_back(charge);
      seedIdx_vec.push_back(seedIdx);
      deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

      // For matching with sim tracks
      hitIdxs.push_back(trk.see_hitIdx()[iSeed][0]);
      hitIdxs.push_back(trk.see_hitIdx()[iSeed][1]);
      hitIdxs.push_back(trk.see_hitIdx()[iSeed][2]);
      char isQuad = false;
      if (trk.see_hitIdx()[iSeed].size() > 3) {
        isQuad = true;
        hitIdxs.push_back(trk.see_hitIdx()[iSeed].size() > 3 ? trk.see_hitIdx()[iSeed][3] : trk.see_hitIdx()[iSeed][2]);
      }
      // if (pt < 0){ ptbin = 0;}
      float neta = 25.;
      float nphi = 72.;
      float nz = 25.;
      int etabin = (p3PCA_Eta + 2.6) / ((2 * 2.6) / neta);
      int phibin = (p3PCA_Phi + 3.14159265358979323846) / ((2. * 3.14159265358979323846) / nphi);
      int dzbin = (trk.see_dz()[iSeed] + 30) / (2 * 30 / nz);
      int isuperbin =
          /*(nz * nphi * neta) * ptbin + (removed since pt bin is determined by pixelType)*/ (nz * nphi) * etabin +
          (nz)*phibin + dzbin;
      // if(isuperbin<0 || isuperbin>=44900){printf("isuperbin %d %d %d %d %f\n",isuperbin,etabin,phibin,dzbin,p3PCA.Eta());}
      superbin_vec.push_back(isuperbin);
      pixelType_vec.push_back(pixtype);
      isQuad_vec.push_back(isQuad);
    }
  }

  out_trkX.push_back(trkX);
  out_trkY.push_back(trkY);
  out_trkZ.push_back(trkZ);
  out_hitId.push_back(hitId);
  out_hitIdxs.push_back(hitIdxs);
  out_hitIndices_vec0.push_back(hitIndices_vec0);
  out_hitIndices_vec1.push_back(hitIndices_vec1);
  out_hitIndices_vec2.push_back(hitIndices_vec2);
  out_hitIndices_vec3.push_back(hitIndices_vec3);
  out_deltaPhi_vec.push_back(deltaPhi_vec);
  out_ptIn_vec.push_back(ptIn_vec);
  out_ptErr_vec.push_back(ptErr_vec);
  out_px_vec.push_back(px_vec);
  out_py_vec.push_back(py_vec);
  out_pz_vec.push_back(pz_vec);
  out_eta_vec.push_back(eta_vec);
  out_etaErr_vec.push_back(etaErr_vec);
  out_phi_vec.push_back(phi_vec);
  out_charge_vec.push_back(charge_vec);
  out_seedIdx_vec.push_back(seedIdx_vec);
  out_superbin_vec.push_back(superbin_vec);
  out_pixelType_vec.push_back(pixelType_vec);
  out_isQuad_vec.push_back(isQuad_vec);

  //    float hit_loading_elapsed = my_timer.RealTime();
  //    if (ana.verbose >= 2) std::cout << "Loading inputs processing time: " << hit_loading_elapsed << " secs" << std::endl;
  //    return hit_loading_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float addInputsToEventPreLoad(LSTEvent *event,
                              bool useOMP,
                              std::vector<float> trkX,
                              std::vector<float> trkY,
                              std::vector<float> trkZ,
                              std::vector<unsigned int> hitId,
                              std::vector<unsigned int> hitIdxs,
                              std::vector<float> ptIn_vec,
                              std::vector<float> ptErr_vec,
                              std::vector<float> px_vec,
                              std::vector<float> py_vec,
                              std::vector<float> pz_vec,
                              std::vector<float> eta_vec,
                              std::vector<float> etaErr_vec,
                              std::vector<float> phi_vec,
                              std::vector<int> charge_vec,
                              std::vector<unsigned int> seedIdx_vec,
                              std::vector<int> superbin_vec,
                              std::vector<PixelType> pixelType_vec,
                              std::vector<char> isQuad_vec) {
  TStopwatch my_timer;

  if (ana.verbose >= 2)
    std::cout << "Loading Inputs (i.e. outer tracker hits, and pixel line segements) to the Line Segment Tracking.... "
              << std::endl;

  my_timer.Start();

  event->addHitToEvent(trkX, trkY, trkZ, hitId, hitIdxs);

  event->addPixelSegmentToEventStart(ptIn_vec,
                                     ptErr_vec,
                                     px_vec,
                                     py_vec,
                                     pz_vec,
                                     eta_vec,
                                     etaErr_vec,
                                     phi_vec,
                                     charge_vec,
                                     seedIdx_vec,
                                     superbin_vec,
                                     pixelType_vec,
                                     isQuad_vec);
  event->wait();  // device side event calls are asynchronous: wait to measure time or print
  float hit_loading_elapsed = my_timer.RealTime();

  if (ana.verbose >= 2)
    std::cout << "Loading inputs processing time: " << hit_loading_elapsed << " secs" << std::endl;

  return hit_loading_elapsed;
}

//________________________________________________________________________________________________________________________________
void printTimingInformation(std::vector<std::vector<float>> &timing_information, float fullTime, float fullavg) {
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
    timing_total += timing[0] * 1000;        // Hits
    timing_total += timing[1] * 1000;        // MD
    timing_total += timing[2] * 1000;        // LS
    timing_total += timing[3] * 1000;        // T3
    timing_total += timing[4] * 1000;        // T5
    timing_total += timing[5] * 1000;        // pLS
    timing_total += timing[6] * 1000;        // pT5
    timing_total += timing[7] * 1000;        // pT3
    timing_total += timing[8] * 1000;        // TC
    timing_total_short += timing[1] * 1000;  // MD
    timing_total_short += timing[2] * 1000;  // LS
    timing_total_short += timing[3] * 1000;  // T3
    timing_total_short += timing[4] * 1000;  // T5
    timing_total_short += timing[6] * 1000;  // pT5
    timing_total_short += timing[7] * 1000;  // pT3
    timing_total_short += timing[8] * 1000;  // TC
    timing_total_short += timing[9] * 1000;  // Reset
    std::cout << std::setw(6) << ievt;
    std::cout << "   " << std::setw(6) << timing[0] * 1000;    // Hits
    std::cout << "   " << std::setw(6) << timing[1] * 1000;    // MD
    std::cout << "   " << std::setw(6) << timing[2] * 1000;    // LS
    std::cout << "   " << std::setw(6) << timing[3] * 1000;    // T3
    std::cout << "   " << std::setw(6) << timing[4] * 1000;    // T5
    std::cout << "   " << std::setw(6) << timing[5] * 1000;    // pLS
    std::cout << "   " << std::setw(6) << timing[6] * 1000;    // pT5
    std::cout << "   " << std::setw(6) << timing[7] * 1000;    // pT3
    std::cout << "   " << std::setw(6) << timing[8] * 1000;    // TC
    std::cout << "   " << std::setw(6) << timing[9] * 1000;    // Reset
    std::cout << "   " << std::setw(7) << timing_total;        // Total time
    std::cout << "   " << std::setw(7) << timing_total_short;  // Total time
    std::cout << std::endl;
    timing_sum_information[0] += timing[0] * 1000;   // Hits
    timing_sum_information[1] += timing[1] * 1000;   // MD
    timing_sum_information[2] += timing[2] * 1000;   // LS
    timing_sum_information[3] += timing[3] * 1000;   // T3
    timing_sum_information[4] += timing[4] * 1000;   // T5
    timing_sum_information[5] += timing[5] * 1000;   // pLS
    timing_sum_information[6] += timing[6] * 1000;   // pT5
    timing_sum_information[7] += timing[7] * 1000;   // pT3
    timing_sum_information[8] += timing[8] * 1000;   // TC
    timing_sum_information[9] += timing[9] * 1000;   // Reset
    timing_shortlist.push_back(timing_total_short);  // short total
    timing_list.push_back(timing_total);             // short total
  }
  timing_sum_information[0] /= timing_information.size();  // Hits
  timing_sum_information[1] /= timing_information.size();  // MD
  timing_sum_information[2] /= timing_information.size();  // LS
  timing_sum_information[3] /= timing_information.size();  // T3
  timing_sum_information[4] /= timing_information.size();  // T5
  timing_sum_information[5] /= timing_information.size();  // pLS
  timing_sum_information[6] /= timing_information.size();  // pT5
  timing_sum_information[7] /= timing_information.size();  // pT3
  timing_sum_information[8] /= timing_information.size();  // TC
  timing_sum_information[9] /= timing_information.size();  // Reset

  float timing_total_avg = 0.0;
  timing_total_avg += timing_sum_information[0];  // Hits
  timing_total_avg += timing_sum_information[1];  // MD
  timing_total_avg += timing_sum_information[2];  // LS
  timing_total_avg += timing_sum_information[3];  // T3
  timing_total_avg += timing_sum_information[4];  // T5
  timing_total_avg += timing_sum_information[5];  // pLS
  timing_total_avg += timing_sum_information[6];  // pT5
  timing_total_avg += timing_sum_information[7];  // pT3
  timing_total_avg += timing_sum_information[8];  // TC
  timing_total_avg += timing_sum_information[9];  // Reset
  float timing_totalshort_avg = 0.0;
  timing_totalshort_avg += timing_sum_information[1];  // MD
  timing_totalshort_avg += timing_sum_information[2];  // LS
  timing_totalshort_avg += timing_sum_information[3];  // T3
  timing_totalshort_avg += timing_sum_information[4];  // T5
  timing_totalshort_avg += timing_sum_information[6];  // pT5
  timing_totalshort_avg += timing_sum_information[7];  // pT3
  timing_totalshort_avg += timing_sum_information[8];  // TC
  timing_totalshort_avg += timing_sum_information[9];  // Reset

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
  std::cout << "   " << std::setw(6) << "pT5";
  std::cout << "   " << std::setw(6) << "pT3";
  std::cout << "   " << std::setw(6) << "TC";
  std::cout << "   " << std::setw(6) << "Reset";
  std::cout << "   " << std::setw(7) << "Total";
  std::cout << "   " << std::setw(7) << "Total(short)";
  std::cout << std::endl;
  std::cout << std::setw(6) << "avg";
  std::cout << "   " << std::setw(6) << timing_sum_information[0];  // Hits
  std::cout << "   " << std::setw(6) << timing_sum_information[1];  // MD
  std::cout << "   " << std::setw(6) << timing_sum_information[2];  // LS
  std::cout << "   " << std::setw(6) << timing_sum_information[3];  // T3
  std::cout << "   " << std::setw(6) << timing_sum_information[4];  // T5
  std::cout << "   " << std::setw(6) << timing_sum_information[5];  // pLS
  std::cout << "   " << std::setw(6) << timing_sum_information[6];  // pT5
  std::cout << "   " << std::setw(6) << timing_sum_information[7];  // pT3
  std::cout << "   " << std::setw(6) << timing_sum_information[8];  // TC
  std::cout << "   " << std::setw(6) << timing_sum_information[9];  // Reset
  std::cout << "   " << std::setw(7) << timing_total_avg;           // Average total time
  std::cout << "   " << std::setw(7) << timing_totalshort_avg;      // Average total time
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
