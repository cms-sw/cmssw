#include <memory>

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboardME11.h"

CSCGEMMotherboardME11::CSCGEMMotherboardME11(unsigned endcap,
                                             unsigned station,
                                             unsigned sector,
                                             unsigned subsector,
                                             unsigned chamber,
                                             const edm::ParameterSet& conf)
    : CSCGEMMotherboard(endcap, station, sector, subsector, chamber, conf)
      // special configuration parameters for ME11 treatment
      ,
      dropLowQualityCLCTsNoGEMs_ME1a_(tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs_ME1a")),
      dropLowQualityCLCTsNoGEMs_ME1b_(tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs_ME1b")),
      dropLowQualityALCTsNoGEMs_ME1a_(tmbParams_.getParameter<bool>("dropLowQualityALCTsNoGEMs_ME1a")),
      dropLowQualityALCTsNoGEMs_ME1b_(tmbParams_.getParameter<bool>("dropLowQualityALCTsNoGEMs_ME1b")),
      buildLCTfromALCTandGEM_ME1a_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM_ME1a")),
      buildLCTfromALCTandGEM_ME1b_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM_ME1b")),
      buildLCTfromCLCTandGEM_ME1a_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM_ME1a")),
      buildLCTfromCLCTandGEM_ME1b_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM_ME1b")),
      promoteCLCTGEMquality_ME1a_(tmbParams_.getParameter<bool>("promoteCLCTGEMquality_ME1a")),
      promoteCLCTGEMquality_ME1b_(tmbParams_.getParameter<bool>("promoteCLCTGEMquality_ME1b")) {
  if (!runPhase2_) {
    edm::LogError("CSCGEMMotherboardME11|SetupError") << "+++ TMB constructed while runPhase2 is not set! +++\n";
  }

  if (!runME11ILT_) {
    edm::LogError("CSCGEMMotherboardME11|SetupError") << "+++ TMB constructed while runME11ILT_ is not set! +++\n";
  };

  // set LUTs
  tmbLUT_ = std::make_unique<CSCGEMMotherboardLUTME11>();
  cscTmbLUT_ = std::make_unique<CSCMotherboardLUTME11>();
}

CSCGEMMotherboardME11::CSCGEMMotherboardME11() : CSCGEMMotherboard() {
  // Constructor used only for testing.
  if (!runPhase2_) {
    edm::LogError("CSCGEMMotherboardME11|SetupError") << "+++ TMB constructed while runPhase2 is not set! +++\n";
  }

  if (!runME11ILT_) {
    edm::LogError("CSCGEMMotherboardME11|SetupError") << "+++ TMB constructed while runME11ILT_ is not set! +++\n";
  }
}

CSCGEMMotherboardME11::~CSCGEMMotherboardME11() {}

//===============================================
//use ALCTs, CLCTs, GEMs to build LCTs
//loop over each BX to find valid ALCT in ME1b, try ALCT-CLCT match, if ALCT-CLCT match failed, try ALCT-GEM match
//do the same in ME1a
//sort LCTs according to different algorithm, and send out number of LCTs up to max_lcts
//===============================================

void CSCGEMMotherboardME11::run(const CSCWireDigiCollection* wiredc,
                                const CSCComparatorDigiCollection* compdc,
                                const GEMPadDigiClusterCollection* gemClusters) {
  CSCGEMMotherboard::clear();
  setupGeometry();
  debugLUTs();

  // encode high multiplicity bits
  unsigned alctBits = alctProc->getHighMultiplictyBits();
  encodeHighMultiplicityBits(alctBits);

  if (gem_g != nullptr) {
    if (infoV >= 0)
      edm::LogInfo("CSCGEMMotherboardME11|SetupInfo") << "+++ run() called for GEM-CSC integrated trigger! +++ \n";
    gemGeometryAvailable = true;
  }

  // check for GEM geometry
  if (not gemGeometryAvailable) {
    edm::LogError("CSCGEMMotherboardME11|SetupError")
        << "+++ run() called for GEM-CSC integrated trigger without valid GEM geometry! +++ \n";
    return;
  }

  if (!(alctProc and clctProc)) {
    edm::LogError("CSCGEMMotherboardME11|SetupError")
        << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alctProc->setCSCGeometry(cscGeometry_);
  clctProc->setCSCGeometry(cscGeometry_);

  alctV = alctProc->run(wiredc);  // run anodeLCT
  clctV = clctProc->run(compdc);  // run cathodeLCT

  processGEMClusters(gemClusters);

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV.empty())
    return;

  int used_clct_mask[20];
  for (int b = 0; b < 20; b++)
    used_clct_mask[b] = 0;

  const bool hasPads(!pads_.empty());
  const bool hasCoPads(hasPads and !coPads_.empty());

  if (debug_matching)
    LogTrace("CSCGEMMotherboardME11") << "hascopads " << hasCoPads << std::endl;

  // ALCT-centric matching
  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
    if (alctProc->getBestALCT(bx_alct).isValid()) {
      const int bx_clct_start(bx_alct - match_trig_window_size / 2 - CSCConstants::ALCT_CLCT_OFFSET);
      const int bx_clct_stop(bx_alct + match_trig_window_size / 2 - CSCConstants::ALCT_CLCT_OFFSET);
      const int bx_copad_start(bx_alct - maxDeltaBXCoPad_);
      const int bx_copad_stop(bx_alct + maxDeltaBXCoPad_);

      if (debug_matching) {
        LogTrace("CSCGEMMotherboardME11")
            << "========================================================================\n"
            << "ALCT-CLCT matching in ME1/1 chamber: " << cscId_ << "\n"
            << "------------------------------------------------------------------------\n"
            << "+++ Best ALCT Details: " << alctProc->getBestALCT(bx_alct) << "\n"
            << "+++ Second ALCT Details: " << alctProc->getSecondALCT(bx_alct) << std::endl;

        printGEMTriggerPads(bx_clct_start, bx_clct_stop, CSCPart::ME11);
        printGEMTriggerCoPads(bx_clct_start, bx_clct_stop, CSCPart::ME11);

        LogTrace("CSCGEMMotherboardME11")
            << "------------------------------------------------------------------------ \n"
            << "Attempt ALCT-CLCT matching in ME1/b in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]"
            << std::endl;
      }

      // ALCT-to-CLCT matching in ME1b
      int nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS)
          continue;
        if (drop_used_clcts and used_clct_mask[bx_clct])
          continue;
        if (clctProc->getBestCLCT(bx_clct).isValid()) {
          if (debug_matching)
            LogTrace("CSCGEMMotherboardME11") << "++Valid ME1b CLCT: " << clctProc->getBestCLCT(bx_clct) << std::endl;

          // pick the pad that corresponds
          matches<GEMPadDigi> mPads;
          matchingPads<GEMPadDigi>(clctProc->getBestCLCT(bx_clct),
                                   clctProc->getSecondCLCT(bx_clct),
                                   alctProc->getBestALCT(bx_alct),
                                   alctProc->getSecondALCT(bx_alct),
                                   mPads);
          matches<GEMCoPadDigi> mCoPads;
          matchingPads<GEMCoPadDigi>(clctProc->getBestCLCT(bx_clct),
                                     clctProc->getSecondCLCT(bx_clct),
                                     alctProc->getBestALCT(bx_alct),
                                     alctProc->getSecondALCT(bx_alct),
                                     mCoPads);

          const bool lowQualityCLCT(clctProc->getBestCLCT(bx_clct).getQuality() <= 3);
          const bool hasMatchingPads(!mPads.empty() or !mCoPads.empty());
          // Low quality LCTs have at most 3 layers. Check if there is a matching GEM pad to keep the CLCT
          if (dropLowQualityCLCTsNoGEMs_ME1b_ and lowQualityCLCT and !hasMatchingPads) {
            LogTrace("CSCGEMMotherboardME11") << "Dropping low quality CLCT without matching GEM pads "
                                              << clctProc->getBestCLCT(bx_clct) << std::endl;
            continue;
          }

          if (debug_matching) {
            LogTrace("CSCGEMMotherboardME11") << "mPads " << mPads.size() << " "
                                              << " mCoPads " << mCoPads.size() << std::endl;
            for (auto p : mPads) {
              LogTrace("CSCGEMMotherboardME11") << p.first << " " << p.second << std::endl;
            }
            for (auto p : mCoPads) {
              LogTrace("CSCGEMMotherboardME11") << p.first << " " << p.second << std::endl;
            }
          }
          //	    if (infoV > 1) LogTrace("CSCMotherboard")
          int mbx = bx_clct - bx_clct_start;
          correlateLCTsGEM(alctProc->getBestALCT(bx_alct),
                           alctProc->getSecondALCT(bx_alct),
                           clctProc->getBestCLCT(bx_clct),
                           clctProc->getSecondCLCT(bx_clct),
                           mPads,
                           mCoPads,
                           allLCTs(bx_alct, mbx, 0),
                           allLCTs(bx_alct, mbx, 1));

          if (allLCTs(bx_alct, mbx, 0).isValid()) {
            ++nSuccesFulMatches;
            used_clct_mask[bx_clct] += 1;

            if (debug_matching) {
              LogTrace("CSCGEMMotherboardME11")
                  << "Successful ALCT-CLCT match in ME1b: bx_alct = " << bx_alct << "; match window: [" << bx_clct_start
                  << "; " << bx_clct_stop << "]; bx_clct = " << bx_clct << "\n"
                  << "+++ Best CLCT Details: " << clctProc->getBestCLCT(bx_clct) << "\n"
                  << "+++ Second CLCT Details: " << clctProc->getSecondCLCT(bx_clct) << std::endl;
              if (allLCTs(bx_alct, mbx, 0).isValid()) {
                LogTrace("CSCGEMMotherboardME11") << "LCT #1 " << allLCTs(bx_alct, mbx, 0) << std::endl;
                LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 0).getALCT() << std::endl;
                LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 0).getCLCT() << std::endl;
                if (allLCTs(bx_alct, mbx, 0).getType() == 2)
                  LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 0).getGEM1() << std::endl;
                if (allLCTs(bx_alct, mbx, 0).getType() == 3)
                  LogTrace("CSCGEMMotherboardME11")
                      << allLCTs(bx_alct, mbx, 0).getGEM1() << " " << allLCTs(bx_alct, mbx, 0).getGEM2() << std::endl;
              }

              if (allLCTs(bx_alct, mbx, 1).isValid()) {
                LogTrace("CSCGEMMotherboardME11") << "LCT #2 " << allLCTs(bx_alct, mbx, 1) << std::endl;
                LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 1).getALCT() << std::endl;
                LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 1).getCLCT() << std::endl;
                if (allLCTs(bx_alct, mbx, 1).getType() == 2)
                  LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 1).getGEM1() << std::endl;
                if (allLCTs(bx_alct, mbx, 1).getType() == 3)
                  LogTrace("CSCGEMMotherboardME11")
                      << allLCTs(bx_alct, mbx, 1).getGEM1() << " " << allLCTs(bx_alct, mbx, 1).getGEM2() << std::endl;
              }
            }

            if (match_earliest_clct_only)
              break;
          } else
            LogTrace("CSCGEMMotherboardME11") << "No valid LCT is built from ALCT-CLCT matching in ME1b" << std::endl;
        }
      }

      // ALCT-to-GEM matching in ME1b
      int nSuccesFulGEMMatches = 0;
      if (nSuccesFulMatches == 0 and buildLCTfromALCTandGEM_ME1b_) {
        if (debug_matching)
          LogTrace("CSCGEMMotherboardME11") << "++No valid ALCT-CLCT matches in ME1b" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }

          // find the best matching copad
          matches<GEMCoPadDigi> copads;
          matchingPads<CSCALCTDigi, GEMCoPadDigi>(
              alctProc->getBestALCT(bx_alct), alctProc->getSecondALCT(bx_alct), copads);

          if (debug_matching)
            LogTrace("CSCGEMMotherboardME11")
                << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : " << copads.size() << std::endl;
          if (copads.empty()) {
            continue;
          }

          CSCGEMMotherboard::correlateLCTsGEM(alctProc->getBestALCT(bx_alct),
                                              alctProc->getSecondALCT(bx_alct),
                                              copads,
                                              allLCTs(bx_alct, 0, 0),
                                              allLCTs(bx_alct, 0, 1));

          if (allLCTs(bx_alct, 0, 0).isValid()) {
            ++nSuccesFulGEMMatches;

            if (debug_matching) {
              LogTrace("CSCGEMMotherboardME11")
                  << "Successful ALCT-GEM CoPad match in ME1b: bx_alct = " << bx_alct << "\n\n"
                  << "------------------------------------------------------------------------" << std::endl
                  << std::endl;
              if (allLCTs(bx_alct, 0, 0).isValid()) {
                LogTrace("CSCGEMMotherboardME11") << "LCT #1 " << allLCTs(bx_alct, 0, 0) << std::endl;
                LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, 0, 0).getALCT() << std::endl;
                if (allLCTs(bx_alct, 0, 0).getType() == 4)
                  LogTrace("CSCGEMMotherboardME11")
                      << allLCTs(bx_alct, 0, 0).getGEM1() << " " << allLCTs(bx_alct, 0, 0).getGEM2() << std::endl
                      << std::endl;
              } else {
                LogTrace("CSCGEMMotherboardME11")
                    << "No valid LCT is built from ALCT-GEM matching in ME1b" << std::endl;
              }
              if (allLCTs(bx_alct, 0, 1).isValid()) {
                LogTrace("CSCGEMMotherboardME11") << "LCT #2 " << allLCTs(bx_alct, 0, 1) << std::endl;
                LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, 0, 1).getALCT() << std::endl;
                if (allLCTs(bx_alct, 0, 1).getType() == 4)
                  LogTrace("CSCGEMMotherboardME11")
                      << allLCTs(bx_alct, 0, 1).getGEM1() << " " << allLCTs(bx_alct, 0, 1).getGEM2() << std::endl
                      << std::endl;
              }
            }

            if (match_earliest_clct_only)
              break;
          }
        }
      }

      if (debug_matching) {
        LogTrace("CSCGEMMotherboardME11")
            << "========================================================================" << std::endl
            << "Summary: " << std::endl;
        if (nSuccesFulMatches > 1)
          LogTrace("CSCGEMMotherboardME11")
              << "Too many successful ALCT-CLCT matches in ME1/1: " << nSuccesFulMatches << ", CSCDetId " << cscId_
              << ", bx_alct = " << bx_alct << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]"
              << std::endl;
        else if (nSuccesFulMatches == 1)
          LogTrace("CSCGEMMotherboardME11")
              << "1 successful ALCT-CLCT match in ME1/1: "
              << " CSCDetId " << cscId_ << ", bx_alct = " << bx_alct << "; match window: [" << bx_clct_start << "; "
              << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulGEMMatches == 1)
          LogTrace("CSCGEMMotherboardME11")
              << "1 successful ALCT-GEM match in ME1/1: "
              << " CSCDetId " << cscId_ << ", bx_alct = " << bx_alct << "; match window: [" << bx_clct_start << "; "
              << bx_clct_stop << "]" << std::endl;
        else
          LogTrace("CSCGEMMotherboardME11") << "Unsuccessful ALCT-CLCT match in ME1/1: "
                                            << "CSCDetId " << cscId_ << ", bx_alct = " << bx_alct << "; match window: ["
                                            << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
      }
    }  // end of ALCT valid block
    else {
      auto coPads(coPads_[bx_alct]);
      if (!coPads.empty()) {
        // keep it simple for the time being, only consider the first copad
        const int bx_clct_start(bx_alct - match_trig_window_size / 2 - CSCConstants::ALCT_CLCT_OFFSET);
        const int bx_clct_stop(bx_alct + match_trig_window_size / 2 - CSCConstants::ALCT_CLCT_OFFSET);

        // matching in ME1b
        if (buildLCTfromCLCTandGEM_ME1b_) {
          for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
            if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS)
              continue;
            if (drop_used_clcts and used_clct_mask[bx_clct])
              continue;
            if (clctProc->getBestCLCT(bx_clct).isValid()) {
              const int quality(clctProc->getBestCLCT(bx_clct).getQuality());
              // only use high-Q stubs for the time being
              if (quality < 4)
                continue;
              int mbx = bx_clct - bx_clct_start;
              CSCGEMMotherboard::correlateLCTsGEM(clctProc->getBestCLCT(bx_clct),
                                                  clctProc->getSecondCLCT(bx_clct),
                                                  coPads,
                                                  allLCTs(bx_alct, mbx, 0),
                                                  allLCTs(bx_alct, mbx, 1));
              if (allLCTs(bx_alct, mbx, 0).isValid()) {
                used_clct_mask[bx_clct] += 1;

                if (debug_matching) {
                  //	    if (infoV > 1) LogTrace("CSCMotherboard")
                  LogTrace("CSCGEMMotherboardME11")
                      << "Successful GEM-CLCT match in ME1b: bx_alct = " << bx_alct << "; match window: ["
                      << bx_clct_start << "; " << bx_clct_stop << "]; bx_clct = " << bx_clct << "\n"
                      << "+++ Best CLCT Details: " << clctProc->getBestCLCT(bx_clct) << "\n"
                      << "+++ Second CLCT Details: " << clctProc->getSecondCLCT(bx_clct) << std::endl;

                  LogTrace("CSCGEMMotherboardME11") << "LCT #1 " << allLCTs(bx_alct, mbx, 0) << std::endl;
                  LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 0).getALCT() << std::endl;
                  if (allLCTs(bx_alct, mbx, 0).getType() == 5)
                    LogTrace("CSCGEMMotherboardME11")
                        << allLCTs(bx_alct, mbx, 0).getGEM1() << " " << allLCTs(bx_alct, mbx, 0).getGEM2() << std::endl
                        << std::endl;

                  LogTrace("CSCGEMMotherboardME11") << "LCT #2 " << allLCTs(bx_alct, mbx, 1) << std::endl;
                  LogTrace("CSCGEMMotherboardME11") << allLCTs(bx_alct, mbx, 1).getALCT() << std::endl;
                  if (allLCTs(bx_alct, mbx, 1).getType() == 5)
                    LogTrace("CSCGEMMotherboardME11")
                        << allLCTs(bx_alct, mbx, 1).getGEM1() << " " << allLCTs(bx_alct, mbx, 1).getGEM2() << std::endl
                        << std::endl;
                }

                if (match_earliest_clct_only)
                  break;
              }
            }
          }  //end of clct loop
        }
      }  // if (!coPads.empty())
    }
  }  // end of ALCT-centric matching

  if (debug_matching) {
    LogTrace("CSCGEMMotherboardME11") << "======================================================================== \n"
                                      << "Counting the final LCTs in CSCGEMMotherboard ME11\n"
                                      << "======================================================================== \n"
                                      << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
    unsigned int n1b = 0, n1a = 0;
    for (const auto& p : readoutLCTs1b()) {
      n1b++;
      LogTrace("CSCGEMMotherboardME11") << "1b LCT " << n1b << "  " << p << std::endl;
    }

    for (const auto& p : readoutLCTs1a()) {
      n1a++;
      LogTrace("CSCGEMMotherboardME11") << "1a LCT " << n1a << "  " << p << std::endl;
    }
  }
}

std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME11::readoutLCTs1a() const { return readoutLCTsME11(ME1A); }

std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME11::readoutLCTs1b() const { return readoutLCTsME11(ME1B); }

// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME11::readoutLCTsME11(enum CSCPart me1ab) const {
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  // The start time of the L1A*LCT coincidence window should be related
  // to the fifo_pretrig parameter, but I am not completely sure how.
  // Just choose it such that the window is centered at bx=7.  This may
  // need further tweaking if the value of tmb_l1a_window_size changes.
  //static int early_tbins = 4;
  // The number of LCT bins in the read-out is given by the
  // tmb_l1a_window_size parameter, forced to be odd
  const int lct_bins = (tmb_l1a_window_size % 2 == 0) ? tmb_l1a_window_size + 1 : tmb_l1a_window_size;
  const int late_tbins = early_tbins + lct_bins;

  // Start from the vector of all found correlated LCTs and select
  // those within the LCT*L1A coincidence window.
  int bx_readout = -1;
  std::vector<CSCCorrelatedLCTDigi> all_lcts;
  switch (tmb_cross_bx_algo) {
    case 0:
    case 1:
      allLCTs.getMatched(all_lcts);
      break;
    case 2:
      sortLCTs(all_lcts, CSCUpgradeMotherboard::sortLCTsByQuality);
      break;
    case 3:
      sortLCTs(all_lcts, CSCUpgradeMotherboard::sortLCTsByGEMDphi);
      break;
    default:
      LogTrace("CSCGEMMotherboardME11") << "tmb_cross_bx_algo error" << std::endl;
      break;
  }

  for (const auto& lct : all_lcts) {
    if (!lct.isValid())
      continue;

    int bx = lct.getBX();
    // Skip LCTs found too early relative to L1Accept.
    if (bx <= early_tbins)
      continue;

    // Skip LCTs found too late relative to L1Accept.
    if (bx > late_tbins)
      continue;

    if ((me1ab == CSCPart::ME1B and lct.getStrip() > CSCConstants::MAX_HALF_STRIP_ME1B) or
        (me1ab == CSCPart::ME1A and lct.getStrip() <= CSCConstants::MAX_HALF_STRIP_ME1B))
      continue;

    // If (readout_earliest_2) take only LCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2 and (bx_readout == -1 or bx == bx_readout)) {
      tmpV.push_back(lct);
      if (bx_readout == -1)
        bx_readout = bx;
    } else
      tmpV.push_back(lct);
  }

  // do a final check on the LCTs in readout
  qualityControl_->checkMultiplicityBX(tmpV);
  for (const auto& lct : tmpV) {
    qualityControl_->checkValid(lct);
  }

  return tmpV;
}

//sort LCTs in each BX
//
void CSCGEMMotherboardME11::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& LCTs,
                                     int bx,
                                     bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const {
  allLCTs.getTimeMatched(bx, LCTs);

  CSCUpgradeMotherboard::sortLCTs(LCTs, *sorter);

  if (LCTs.size() > max_lcts)
    LCTs.erase(LCTs.begin() + max_lcts, LCTs.end());
}

//sort LCTs in whole LCTs BX window
void CSCGEMMotherboardME11::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& LCTs,
                                     bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const {
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    // temporary collection with all LCTs in the whole chamber
    std::vector<CSCCorrelatedLCTDigi> LCTs_tmp;
    CSCGEMMotherboardME11::sortLCTs(LCTs_tmp, bx, *sorter);

    // Add the LCTs
    for (const auto& p : LCTs_tmp) {
      LCTs.push_back(p);
    }
  }
}

bool CSCGEMMotherboardME11::doesALCTCrossCLCT(const CSCALCTDigi& a, const CSCCLCTDigi& c) const {
  return cscTmbLUT_->doesALCTCrossCLCT(a, c, theEndcap, gangedME1a_);
}

bool CSCGEMMotherboardME11::doesWiregroupCrossStrip(int key_wg, int key_strip) const {
  return cscTmbLUT_->doesWiregroupCrossStrip(key_wg, key_strip, theEndcap, gangedME1a_);
}

void CSCGEMMotherboardME11::correlateLCTsGEM(const CSCALCTDigi& bALCT,
                                             const CSCALCTDigi& sALCT,
                                             const CSCCLCTDigi& bCLCT,
                                             const CSCCLCTDigi& sCLCT,
                                             const GEMPadDigiIds& pads,
                                             const GEMCoPadDigiIds& copads,
                                             CSCCorrelatedLCTDigi& lct1,
                                             CSCCorrelatedLCTDigi& lct2) const {
  const CSCALCTDigi& bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  const CSCCLCTDigi& bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

  // assume that always anodeBestValid and cathodeBestValid
  if (secondALCT == bestALCT)
    secondALCT.clear();
  if (secondCLCT == bestCLCT)
    secondCLCT.clear();

  const bool ok_bb = bestALCT.isValid() and bestCLCT.isValid();
  const bool ok_bs = bestALCT.isValid() and secondCLCT.isValid();
  const bool ok_sb = secondALCT.isValid() and bestCLCT.isValid();
  const bool ok_ss = secondALCT.isValid() and secondCLCT.isValid();

  const int ok11 = doesALCTCrossCLCT(bestALCT, bestCLCT);
  const int ok12 = doesALCTCrossCLCT(bestALCT, secondCLCT);
  const int ok21 = doesALCTCrossCLCT(secondALCT, bestCLCT);
  const int ok22 = doesALCTCrossCLCT(secondALCT, secondCLCT);
  const int code = (ok11 << 3) | (ok12 << 2) | (ok21 << 1) | (ok22);

  int dbg = 0;
  if (dbg)
    LogTrace("CSCGEMMotherboardME11") << "debug correlateLCTs in ME11" << cscId_ << "\n"
                                      << "ALCT1: " << bestALCT << "\n"
                                      << "ALCT2: " << secondALCT << "\n"
                                      << "CLCT1: " << bestCLCT << "\n"
                                      << "CLCT2: " << secondCLCT << "\n"
                                      << "ok 11 12 21 22 code = " << ok11 << " " << ok12 << " " << ok21 << " " << ok22
                                      << " " << code << std::endl;

  if (code == 0)
    return;

  // LUT defines correspondence between possible ok## combinations
  // and resulting lct1 and lct2
  int lut[16][2] = {
      //ok: 11 12 21 22
      {0, 0},    // 0  0  0  0
      {22, 0},   // 0  0  0  1
      {21, 0},   // 0  0  1  0
      {21, 22},  // 0  0  1  1
      {12, 0},   // 0  1  0  0
      {12, 22},  // 0  1  0  1
      {12, 21},  // 0  1  1  0
      {12, 21},  // 0  1  1  1
      {11, 0},   // 1  0  0  0
      {11, 22},  // 1  0  0  1
      {11, 21},  // 1  0  1  0
      {11, 22},  // 1  0  1  1
      {11, 12},  // 1  1  0  0
      {11, 22},  // 1  1  0  1
      {11, 12},  // 1  1  1  0
      {11, 22},  // 1  1  1  1
  };

  if (dbg)
    LogTrace("CSCGEMMotherboardME11") << "lut 0 1 = " << lut[code][0] << " " << lut[code][1] << std::endl;

  // check matching copads
  const GEMCoPadDigi& bb_copad = bestMatchingPad<GEMCoPadDigi>(bestALCT, bestCLCT, copads);
  const GEMCoPadDigi& bs_copad = bestMatchingPad<GEMCoPadDigi>(bestALCT, secondCLCT, copads);
  const GEMCoPadDigi& sb_copad = bestMatchingPad<GEMCoPadDigi>(secondALCT, bestCLCT, copads);
  const GEMCoPadDigi& ss_copad = bestMatchingPad<GEMCoPadDigi>(secondALCT, secondCLCT, copads);

  // check matching pads
  const GEMPadDigi& bb_pad = bestMatchingPad<GEMPadDigi>(bestALCT, bestCLCT, pads);
  const GEMPadDigi& bs_pad = bestMatchingPad<GEMPadDigi>(bestALCT, secondCLCT, pads);
  const GEMPadDigi& sb_pad = bestMatchingPad<GEMPadDigi>(secondALCT, bestCLCT, pads);
  const GEMPadDigi& ss_pad = bestMatchingPad<GEMPadDigi>(secondALCT, secondCLCT, pads);

  // evaluate possible combinations
  const bool ok_bb_copad = ok11 == 1 and ok_bb and bb_copad.isValid();
  const bool ok_bs_copad = ok12 == 1 and ok_bs and bs_copad.isValid();
  const bool ok_sb_copad = ok21 == 1 and ok_sb and sb_copad.isValid();
  const bool ok_ss_copad = ok22 == 1 and ok_ss and ss_copad.isValid();

  const bool ok_bb_pad = ok11 == 1 and ok_bb and bb_pad.isValid();
  const bool ok_bs_pad = ok12 == 1 and ok_bs and bs_pad.isValid();
  const bool ok_sb_pad = ok21 == 1 and ok_sb and sb_pad.isValid();
  const bool ok_ss_pad = ok22 == 1 and ok_ss and ss_pad.isValid();

  switch (lut[code][0]) {
    case 11:
      if (ok_bb_copad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, bestCLCT, bb_copad, 1);
      else if (ok_bb_pad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, bestCLCT, bb_pad, 1);
      else
        lct1 = constructLCTs(bestALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
      break;
    case 12:
      if (ok_bs_copad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_copad, 1);
      else if (ok_bs_pad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_pad, 1);
      else
        lct1 = constructLCTs(bestALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
      break;
    case 21:
      if (ok_sb_copad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_copad, 1);
      else if (ok_sb_pad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_pad, 1);
      else
        lct1 = constructLCTs(secondALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
      break;
    case 22:
      if (ok_ss_copad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, ss_copad, 1);
      else if (ok_ss_pad)
        lct1 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, ss_pad, 1);
      else
        lct1 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
      break;
    default:
      return;
  }

  if (dbg)
    LogTrace("CSCGEMMotherboardME11") << "lct1: " << lct1 << std::endl;

  switch (lut[code][1]) {
    case 12:
      if (ok_bs_copad)
        lct2 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_copad, 2);
      else if (ok_bs_pad)
        lct2 = CSCGEMMotherboard::constructLCTsGEM(bestALCT, secondCLCT, bs_pad, 2);
      else
        lct2 = constructLCTs(bestALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
      break;
    case 21:
      if (ok_sb_copad)
        lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_copad, 2);
      else if (ok_sb_pad)
        lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, bestCLCT, sb_pad, 2);
      else
        lct2 = constructLCTs(secondALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
      break;
    case 22:
      if (ok_bb_copad)
        lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, bb_copad, 2);
      else if (ok_bb_pad)
        lct2 = CSCGEMMotherboard::constructLCTsGEM(secondALCT, secondCLCT, bb_pad, 2);
      else
        lct2 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
      break;
    default:
      return;
  }

  if (dbg)
    LogTrace("CSCGEMMotherboardME11") << "lct2: " << lct2 << std::endl;

  if (dbg)
    LogTrace("CSCGEMMotherboardME11") << "out of correlateLCTs" << std::endl;
  return;
}
