#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeCathodeLCTProcessor.h"

#include <iomanip>

CSCUpgradeCathodeLCTProcessor::CSCUpgradeCathodeLCTProcessor(unsigned endcap,
                                                             unsigned station,
                                                             unsigned sector,
                                                             unsigned subsector,
                                                             unsigned chamber,
                                                             const edm::ParameterSet& conf)
    : CSCCathodeLCTProcessor(endcap, station, sector, subsector, chamber, conf) {
  if (!runPhase2_)
    edm::LogError("CSCUpgradeCathodeLCTProcessor|ConfigError")
        << "+++ Upgrade CSCUpgradeCathodeLCTProcessor constructed while runPhase2_ is not set! +++\n";

  // use of localized dead-time zones
  use_dead_time_zoning_ = clctParams_.getParameter<bool>("useDeadTimeZoning");
  clct_state_machine_zone_ = clctParams_.getParameter<unsigned int>("clctStateMachineZone");

  // how far away may trigger happen from pretrigger
  pretrig_trig_zone_ = clctParams_.getParameter<unsigned int>("clctPretriggerTriggerZone");
}

// --------------------------------------------------------------------------
// The code below is for Phase2 studies of the CLCT algorithm
// --------------------------------------------------------------------------

// Phase2 version, add the feature of localized dead time zone for pretrigger
bool CSCUpgradeCathodeLCTProcessor::preTrigger(const int start_bx, int& first_bx) {
  if (runPhase2_ and !use_dead_time_zoning_) {
    return CSCCathodeLCTProcessor::preTrigger(start_bx, first_bx);
  }

  if (infoV > 1)
    LogTrace("CSCUpgradeCathodeLCTProcessor")
        << "....................PreTrigger, Phase2 version with localized dead time zone...........................";

  int nPreTriggers = 0;

  bool pre_trig = false;

  // Now do a loop over bx times to see (if/when) track goes over threshold
  for (unsigned int bx_time = start_bx; bx_time < fifo_tbins; bx_time++) {
    // For any given bunch-crossing, start at the lowest keystrip and look for
    // the number of separate layers in the pattern for that keystrip that have
    // pulses at that bunch-crossing time.  Do the same for the next keystrip,
    // etc.  Then do the entire process again for the next bunch-crossing, etc
    // until you find a pre-trigger.
    std::map<int, std::map<int, CSCCLCTDigi::ComparatorContainer> > hits_in_patterns;
    hits_in_patterns.clear();

    bool hits_in_time = patternFinding(bx_time, hits_in_patterns);
    if (hits_in_time) {
      for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
        if (infoV > 1) {
          if (nhits[hstrip] > 0) {
            LogTrace("CSCUpgradeCathodeLCTProcessor")
                << " bx = " << std::setw(2) << bx_time << " --->"
                << " halfstrip = " << std::setw(3) << hstrip << " best pid = " << std::setw(2) << best_pid[hstrip]
                << " nhits = " << nhits[hstrip];
          }
        }
        // note that ispretrig_ is initialized in findLCT function
        if (nhits[hstrip] >= nplanes_hit_pretrig && best_pid[hstrip] >= pid_thresh_pretrig &&
            !busyMap_[hstrip][bx_time]) {
          pre_trig = true;
          ispretrig_[hstrip] = true;

          // write each pre-trigger to output
          nPreTriggers++;
          thePreTriggerDigis.push_back(constructPreCLCT(bx_time, hstrip, nPreTriggers));
        }
        // busy zone, keep pretriggering, ignore this
        else if (nhits[hstrip] >= nplanes_hit_pretrig && best_pid[hstrip] >= pid_thresh_pretrig) {
          ispretrig_[hstrip] = true;
          if (infoV > 1)
            LogTrace("CSCUpgradeCathodeLCTProcessor")
                << " halfstrip " << std::setw(3) << hstrip << " in dead zone and is pretriggerred";
        }
        // no pretrigger on this halfstrip, release dead zone
        else if (nhits[hstrip] < nplanes_hit_pretrig || best_pid[hstrip] < pid_thresh_pretrig) {
          ispretrig_[hstrip] = false;
        }
      }  // find all pretriggers

      // update dead zone
      markBusyZone(bx_time);

      if (pre_trig) {
        first_bx = bx_time;  // bx at time of pretrigger
        return true;
      }
    } else {
      // no pattern found, remove all dead zone
      clearPreTriggers();
    }
  }  // end loop over bx times

  if (infoV > 1)
    LogTrace("CSCUpgradeCathodeLCTProcessor") << "no pretrigger, returning \n";
  first_bx = fifo_tbins;
  return false;
}  // preTrigger -- Phase2 version.

// Phase2 version.
std::vector<CSCCLCTDigi> CSCUpgradeCathodeLCTProcessor::findLCTs(
    const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) {
  // run the original algorithm in case we do not use dead time zoning
  if (runPhase2_ and !use_dead_time_zoning_) {
    return CSCCathodeLCTProcessor::findLCTs(halfstrip);
  }

  std::vector<CSCCLCTDigi> lctList;

  // initialize the ispretrig_ before doing pretriggering
  clearPreTriggers();

  if (infoV > 1)
    dumpDigis(halfstrip);

  // keeps dead-time zones around key halfstrips of triggered CLCTs
  for (int i = 0; i < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER; i++) {
    for (int j = 0; j < CSCConstants::MAX_CLCT_TBINS; j++) {
      busyMap_[i][j] = false;
    }
  }

  // Fire half-strip one-shots for hit_persist bx's (4 bx's by default).
  pulseExtension(halfstrip);

  unsigned int start_bx = start_bx_shift;
  // Stop drift_delay bx's short of fifo_tbins since at later bx's we will
  // not have a full set of hits to start pattern search anyway.
  unsigned int stop_bx = fifo_tbins - drift_delay;

  // Allow for more than one pass over the hits in the time window.
  // Do search in every BX
  while (start_bx < stop_bx) {
    // temp CLCT objects
    CSCCLCTDigi tempBestCLCT;
    CSCCLCTDigi tempSecondCLCT;

    // All half-strip pattern envelopes are evaluated simultaneously, on every clock cycle.
    int first_bx = 999;

    // Check for a pre-trigger. If so, find the first BX when the pre-trigger occurred
    bool pre_trig = CSCUpgradeCathodeLCTProcessor::preTrigger(start_bx, first_bx);

    // If any of half-strip envelopes has enough layers hit in it, TMB
    // will pre-trigger.
    if (pre_trig) {
      if (infoV > 1)
        LogTrace("CSCUpgradeCathodeLCTProcessor")
            << "..... pretrigger at bx = " << first_bx << "; waiting drift delay .....";

      // TMB latches LCTs drift_delay clocks after pretrigger.
      int latch_bx = first_bx + drift_delay;

      // temporary container to keep track of the hits in the CLCT
      std::map<int, std::map<int, CSCCLCTDigi::ComparatorContainer> > hits_in_patterns;
      hits_in_patterns.clear();

      // We check if there is at least one key half strip for which at least
      // one pattern id has at least the minimum number of hits
      bool hits_in_time = patternFinding(latch_bx, hits_in_patterns);
      if (infoV > 1) {
        if (hits_in_time) {
          for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
            if (nhits[hstrip] > 0) {
              LogTrace("CSCUpgradeCathodeLCTProcessor")
                  << " bx = " << std::setw(2) << latch_bx << " --->"
                  << " halfstrip = " << std::setw(3) << hstrip << " best pid = " << std::setw(2) << best_pid[hstrip]
                  << " nhits = " << nhits[hstrip];
            }
          }
        }
      }

      // Quality for sorting.
      int quality[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER];
      int best_halfstrip[CSCConstants::MAX_CLCTS_PER_PROCESSOR], best_quality[CSCConstants::MAX_CLCTS_PER_PROCESSOR];
      for (int ilct = 0; ilct < CSCConstants::MAX_CLCTS_PER_PROCESSOR; ilct++) {
        best_halfstrip[ilct] = -1;
        best_quality[ilct] = 0;
      }

      bool pretrig_zone[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER];

      // Calculate quality from pattern id and number of hits, and
      // simultaneously select best-quality CLCT.
      if (hits_in_time) {
        // first, mark half-strip zones around pretriggers
        // that happened at the current first_bx
        markPreTriggerZone(pretrig_zone);

        for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
          /* The bend-direction bit pid[0] is ignored (left and right bends have equal quality).
           This works both for the Run-2 patterns
           - PID 2,3: 2 & 14 == 2, 3 & 14 == 2
           - PID 4,5: 4 & 14 == 4, 3 & 14 == 4
           - PID 6,7: 6 & 14 == 6, 3 & 14 == 6
           - PID 8,9: 8 & 14 == 8, 3 & 14 == 8
           - PID 10: 10 & 14 == 10
           It also works for the Run-3 patterns:
           - PID 0,1: 0 & 14 == 0, 1 & 14 == 0
           - PID 2,3: 2 & 14 == 2, 3 & 14 == 2
           - PID  4:  4 & 14 == 4
          */
          quality[hstrip] = (best_pid[hstrip] & 14) | (nhits[hstrip] << 5);
          // do not consider halfstrips:
          // - out of pretrigger-trigger zones
          // - in busy zones from previous trigger
          if (quality[hstrip] > best_quality[0] && pretrig_zone[hstrip] && !busyMap_[hstrip][first_bx]) {
            best_halfstrip[0] = hstrip;
            best_quality[0] = quality[hstrip];
            // temporary alias
            const int best_hs(best_halfstrip[0]);
            const int best_pat(best_pid[best_hs]);
            // construct a CLCT if the trigger condition has been met
            if (best_hs >= 0 && nhits[best_hs] >= nplanes_hit_pattern) {
              // overwrite the current best CLCT
              tempBestCLCT = constructCLCT(first_bx, best_hs, hits_in_patterns[best_hs][best_pat]);
            }
          }
        }
      }

      // If 1st best CLCT is found, look for the 2nd best.
      if (best_halfstrip[0] >= 0) {
        // Get the half-strip of the best CLCT in this BX that was put into the list.
        // You do need to re-add the any stagger, because the busy keys are based on
        // the pulse array which takes into account strip stagger!!!
        const unsigned halfStripBestCLCT(tempBestCLCT.getKeyStrip() + stagger[CSCConstants::KEY_CLCT_LAYER - 1]);

        // Mark keys near best CLCT as busy by setting their quality to
        // zero, and repeat the search.
        //markBusyKeys(best_halfstrip[0], best_pid[best_halfstrip[0]], quality);
        markBusyKeys(halfStripBestCLCT, best_pid[halfStripBestCLCT], quality);

        for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
          // we don't have to recalculate the quality for each half-strip
          if (quality[hstrip] > best_quality[1] && pretrig_zone[hstrip] && !busyMap_[hstrip][first_bx]) {
            best_halfstrip[1] = hstrip;
            best_quality[1] = quality[hstrip];
            // temporary alias
            const int best_hs(best_halfstrip[1]);
            const int best_pat(best_pid[best_hs]);
            // construct a CLCT if the trigger condition has been met
            if (best_hs >= 0 && nhits[best_hs] >= nplanes_hit_pattern) {
              // overwrite the current second best CLCT
              tempSecondCLCT = constructCLCT(first_bx, best_hs, hits_in_patterns[best_hs][best_pat]);
            }
          }
        }

        // Sort bestCLCT and secondALCT by quality
        // if qualities are the same, sort by run-2 or run-3 pattern
        // if qualities and patterns are the same, sort by half strip number
        bool changeOrder = false;

        unsigned qualityBest = 0, qualitySecond = 0;
        unsigned patternBest = 0, patternSecond = 0;
        unsigned halfStripBest = 0, halfStripSecond = 0;

        if (tempBestCLCT.isValid() and tempSecondCLCT.isValid()) {
          qualityBest = tempBestCLCT.getQuality();
          qualitySecond = tempSecondCLCT.getQuality();
          if (!run3_) {
            patternBest = tempBestCLCT.getPattern();
            patternSecond = tempSecondCLCT.getPattern();
          } else {
            patternBest = tempBestCLCT.getRun3Pattern();
            patternSecond = tempSecondCLCT.getRun3Pattern();
          }
          halfStripBest = tempBestCLCT.getKeyStrip();
          halfStripSecond = tempSecondCLCT.getKeyStrip();

          if (qualitySecond > qualityBest)
            changeOrder = true;
          else if ((qualitySecond == qualityBest) and (int(patternSecond / 2) > int(patternBest / 2)))
            changeOrder = true;
          else if ((qualitySecond == qualityBest) and (int(patternSecond / 2) == int(patternBest / 2)) and
                   (halfStripSecond < halfStripBest))
            changeOrder = true;
        }

        CSCCLCTDigi tempCLCT;
        if (changeOrder) {
          tempCLCT = tempBestCLCT;
          tempBestCLCT = tempSecondCLCT;
          tempSecondCLCT = tempCLCT;
        }

        // add the CLCTs to the collection
        if (tempBestCLCT.isValid()) {
          lctList.push_back(tempBestCLCT);
        }
        if (tempSecondCLCT.isValid()) {
          lctList.push_back(tempSecondCLCT);
        }
      }  //find CLCT, end of best_halfstrip[0] >= 0
    }    //pre_trig
    // The pattern finder runs continuously, so another pre-trigger
    // could occur already at the next bx.
    start_bx = first_bx + 1;
  }
  return lctList;
}  // findLCTs -- Phase2 version.

void CSCUpgradeCathodeLCTProcessor::markPreTriggerZone(
    bool pretrig_zone[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) const {
  // first reset the pretrigger zone (no pretriggers anywhere in this BX
  for (int hstrip = 0; hstrip < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER; hstrip++) {
    pretrig_zone[hstrip] = false;
  }
  // then set the pretrigger zone according to the ispretrig_ array
  for (int hstrip = 0; hstrip < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER; hstrip++) {
    if (ispretrig_[hstrip]) {
      int min_hs = hstrip - pretrig_trig_zone_;
      int max_hs = hstrip + pretrig_trig_zone_;
      // set the minimum strip
      if (min_hs < 0)
        min_hs = 0;
      // set the maximum strip
      if (max_hs > CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER - 1)
        max_hs = CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER - 1;
      // mark the pre-trigger zone
      for (int hs = min_hs; hs <= max_hs; hs++)
        pretrig_zone[hs] = true;
      if (infoV > 1)
        LogTrace("CSCUpgradeCathodeLCTProcessor")
            << " marked pretrigger halfstrip zone [" << min_hs << "," << max_hs << "]";
    }
  }
}

void CSCUpgradeCathodeLCTProcessor::markBusyZone(const int bx) {
  for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
    // check if this halfstrip has a pretrigger
    if (ispretrig_[hstrip]) {
      // only fixed localized dead time zone is implemented in firmware
      int min_hstrip = hstrip - clct_state_machine_zone_;
      int max_hstrip = hstrip + clct_state_machine_zone_;
      // set the minimum strip
      if (min_hstrip < stagger[CSCConstants::KEY_CLCT_LAYER - 1])
        min_hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1];
      // set the maximum strip
      if (max_hstrip >= numHalfStrips_)
        max_hstrip = numHalfStrips_ - 1;
      // mask the busy half-strips for 1 BX after the pretrigger
      for (int hs = min_hstrip; hs <= max_hstrip; hs++)
        busyMap_[hs][bx + 1] = true;
      if (infoV > 1)
        LogTrace("CSCUpgradeCathodeLCTProcessor")
            << " marked zone around pretriggerred halfstrip " << hstrip << " as dead zone for pretriggering at bx"
            << bx + 1 << " halfstrip: [" << min_hstrip << "," << max_hstrip << "]";
    }
  }
}
