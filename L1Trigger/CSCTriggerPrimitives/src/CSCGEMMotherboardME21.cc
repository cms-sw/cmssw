#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboardME21.h"

CSCGEMMotherboardME21::CSCGEMMotherboardME21(unsigned endcap,
                                             unsigned station,
                                             unsigned sector,
                                             unsigned subsector,
                                             unsigned chamber,
                                             const edm::ParameterSet& conf)
    : CSCGEMMotherboard(endcap, station, sector, subsector, chamber, conf),
      dropLowQualityCLCTsNoGEMs_(tmbParams_.getParameter<bool>("dropLowQualityCLCTsNoGEMs")),
      dropLowQualityALCTsNoGEMs_(tmbParams_.getParameter<bool>("dropLowQualityALCTsNoGEMs")),
      buildLCTfromALCTandGEM_(tmbParams_.getParameter<bool>("buildLCTfromALCTandGEM")),
      buildLCTfromCLCTandGEM_(tmbParams_.getParameter<bool>("buildLCTfromCLCTandGEM")) {
  if (!isSLHC_ or !runME21ILT_)
    edm::LogError("CSCGEMMotherboardME21|ConfigError")
        << "+++ Upgrade CSCGEMMotherboardME21 constructed while isSLHC is not set! +++\n";

  // set LUTs
  tmbLUT_.reset(new CSCGEMMotherboardLUTME21());
}

CSCGEMMotherboardME21::CSCGEMMotherboardME21() : CSCGEMMotherboard() {
  if (!isSLHC_ or !runME21ILT_)
    edm::LogError("CSCGEMMotherboardME21|ConfigError")
        << "+++ Upgrade CSCGEMMotherboardME21 constructed while isSLHC is not set! +++\n";
}

CSCGEMMotherboardME21::~CSCGEMMotherboardME21() {}

void CSCGEMMotherboardME21::run(const CSCWireDigiCollection* wiredc,
                                const CSCComparatorDigiCollection* compdc,
                                const GEMPadDigiClusterCollection* gemClusters) {
  std::unique_ptr<GEMPadDigiCollection> gemPads(new GEMPadDigiCollection());
  coPadProcessor->declusterize(gemClusters, *gemPads);
  run(wiredc, compdc, gemPads.get());
}

void CSCGEMMotherboardME21::run(const CSCWireDigiCollection* wiredc,
                                const CSCComparatorDigiCollection* compdc,
                                const GEMPadDigiCollection* gemPads) {
  CSCGEMMotherboard::clear();
  setupGeometry();
  debugLUTs();

  //  generator_->generateLUTs(theEndcap, theStation, theSector, theSubsector, theTrigChamber);

  if (gem_g != nullptr) {
    if (infoV >= 0)
      edm::LogInfo("CSCGEMMotherboardME21|SetupInfo") << "+++ run() called for GEM-CSC integrated trigger! +++ \n";
    gemGeometryAvailable = true;
  }

  // check for GEM geometry
  if (not gemGeometryAvailable) {
    if (infoV >= 0)
      edm::LogError("CSCGEMMotherboardME21|SetupError")
          << "+++ run() called for GEM-CSC integrated trigger without valid GEM geometry! +++ \n";
    return;
  }
  gemCoPadV = coPadProcessor->run(gemPads);  // run copad processor in GE1/1

  if (!(alctProc and clctProc)) {
    if (infoV >= 0)
      edm::LogError("CSCGEMMotherboardME21|SetupError")
          << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alctProc->setCSCGeometry(cscGeometry_);
  clctProc->setCSCGeometry(cscGeometry_);

  alctV = alctProc->run(wiredc);  // run anodeLCT
  clctV = clctProc->run(compdc);  // run cathodeLCT

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV.empty())
    return;

  LogTrace("CSCGEMCMotherboardME21") << "ALL ALCTs from ME21 " << std::endl;
  for (const auto& alct : alctV)
    if (alct.isValid())
      LogTrace("CSCGEMCMotherboardME21") << alct << std::endl;

  LogTrace("CSCGEMCMotherboardME21") << "ALL CLCTs from ME21 " << std::endl;
  for (const auto& clct : clctV)
    if (clct.isValid())
      LogTrace("CSCGEMCMotherboardME21") << clct << std::endl;

  int used_clct_mask[20];
  for (int c = 0; c < 20; ++c)
    used_clct_mask[c] = 0;

  // retrieve pads and copads in a certain BX window for this CSC

  retrieveGEMPads(gemPads, gemId);
  retrieveGEMCoPads();

  const bool hasCoPads(!coPads_.empty());

  // ALCT centric matching
  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
    if (alctProc->getBestALCT(bx_alct).isValid()) {
      const int bx_clct_start(bx_alct - match_trig_window_size / 2 - alctClctOffset_);
      const int bx_clct_stop(bx_alct + match_trig_window_size / 2 - alctClctOffset_);
      const int bx_copad_start(bx_alct - maxDeltaBXCoPad_);
      const int bx_copad_stop(bx_alct + maxDeltaBXCoPad_);

      if (debug_matching) {
        LogTrace("CSCGEMMotherboardME21")
            << "========================================================================" << std::endl
            << "ALCT-CLCT matching in ME2/1 chamber: " << cscId_ << " in bx range: [" << bx_clct_start << ","
            << bx_clct_stop << "] for bx " << bx_alct << std::endl
            << "------------------------------------------------------------------------" << std::endl
            << "+++ Best ALCT Details: " << alctProc->getBestALCT(bx_alct) << std::endl
            << "+++ Second ALCT Details: " << alctProc->getSecondALCT(bx_alct) << std::endl;

        printGEMTriggerPads(bx_clct_start, bx_clct_stop, CSCPart::ME21);
        printGEMTriggerCoPads(bx_clct_start, bx_clct_stop, CSCPart::ME21);
      }

      // ALCT-to-CLCT
      int nGoodMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS)
          continue;
        if (drop_used_clcts and used_clct_mask[bx_clct])
          continue;
        if (clctProc->getBestCLCT(bx_clct).isValid()) {
          if (debug_matching) {
            LogTrace("CSCGEMMotherboardME21")
                << "+++ Best CLCT Details: " << clctProc->getBestCLCT(bx_clct) << std::endl
                << "+++ Second CLCT Details: " << clctProc->getSecondCLCT(bx_clct) << std::endl;
          }

          // clct quality
          const bool lowQualityCLCT(clctProc->getBestCLCT(bx_clct).getQuality() <= 3);
          // low quality ALCT
          const bool lowQualityALCT(alctProc->getBestALCT(bx_alct).getQuality() == 0);

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

          bool hasMatchingPads(!mPads.empty() or !mCoPads.empty());

          if (dropLowQualityCLCTsNoGEMs_ and lowQualityCLCT and !hasMatchingPads) {
            continue;
          }
          if (dropLowQualityALCTsNoGEMs_ and lowQualityALCT and !hasMatchingPads) {
            continue;
          }

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
            used_clct_mask[bx_clct] += 1;
            ++nGoodMatches;

            if (debug_matching) {
              LogTrace("CSCGEMMotherboardME21")
                  << "Good ALCT-CLCT match in ME21: bx_alct = " << bx_alct << "; match window: [" << bx_clct_start
                  << "; " << bx_clct_stop << "]; bx_clct = " << bx_clct << "\n"
                  << std::endl;

              if (allLCTs(bx_alct, mbx, 0).isValid()) {
                LogTrace("CSCGEMMotherboardME21") << "LCT #1 " << allLCTs(bx_alct, mbx, 0) << std::endl
                                                  << allLCTs(bx_alct, mbx, 0).getALCT() << std::endl
                                                  << allLCTs(bx_alct, mbx, 0).getCLCT() << std::endl;
                if (allLCTs(bx_alct, mbx, 0).getType() == 2)
                  LogTrace("CSCGEMMotherboardME21") << allLCTs(bx_alct, mbx, 0).getGEM1() << std::endl;
                if (allLCTs(bx_alct, mbx, 0).getType() == 3)
                  LogTrace("CSCGEMMotherboardME21")
                      << allLCTs(bx_alct, mbx, 0).getGEM1() << " " << allLCTs(bx_alct, mbx, 0).getGEM2() << std::endl;
              }

              if (allLCTs(bx_alct, mbx, 1).isValid()) {
                LogTrace("CSCGEMMotherboardME21") << "LCT #2 " << allLCTs(bx_alct, mbx, 1) << std::endl
                                                  << allLCTs(bx_alct, mbx, 1).getALCT() << std::endl
                                                  << allLCTs(bx_alct, mbx, 1).getCLCT() << std::endl;
                if (allLCTs(bx_alct, mbx, 1).getType() == 2)
                  LogTrace("CSCGEMMotherboardME21") << allLCTs(bx_alct, mbx, 1).getGEM1() << std::endl;
                if (allLCTs(bx_alct, mbx, 1).getType() == 3)
                  LogTrace("CSCGEMMotherboardME21")
                      << allLCTs(bx_alct, mbx, 1).getGEM1() << " " << allLCTs(bx_alct, mbx, 1).getGEM2() << std::endl;
              }
            }

            if (match_earliest_clct_only) {
              break;
            }
          }
        }
      }

      // ALCT-to-GEM matching
      int nGoodGEMMatches = 0;
      if (nGoodMatches == 0 and buildLCTfromALCTandGEM_) {
        if (debug_matching)
          LogTrace("CSCGEMMotherboardME21") << "++No valid ALCT-CLCT matches in ME21" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }

          // find the best matching copad
          matches<GEMCoPadDigi> copads;
          matchingPads<CSCALCTDigi, GEMCoPadDigi>(
              alctProc->getBestALCT(bx_alct), alctProc->getSecondALCT(bx_alct), copads);

          if (debug_matching)
            LogTrace("CSCGEMMotherboardME21")
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
            ++nGoodGEMMatches;

            if (debug_matching) {
              LogTrace("CSCGEMMotherboardME21")
                  << "Good ALCT-GEM CoPad match in ME21: bx_alct = " << bx_alct << "\n\n"
                  << "------------------------------------------------------------------------" << std::endl
                  << std::endl;
              if (allLCTs(bx_alct, 0, 0).isValid()) {
                LogTrace("CSCGEMMotherboardME21") << "LCT #1 " << allLCTs(bx_alct, 0, 0) << std::endl
                                                  << allLCTs(bx_alct, 0, 0).getALCT() << std::endl;
                if (allLCTs(bx_alct, 0, 0).getType() == 4)
                  LogTrace("CSCGEMMotherboardME21")
                      << allLCTs(bx_alct, 0, 0).getGEM1() << " " << allLCTs(bx_alct, 0, 0).getGEM2() << std::endl
                      << std::endl;
              }
              if (allLCTs(bx_alct, 0, 1).isValid()) {
                LogTrace("CSCGEMMotherboardME21") << "LCT #2 " << allLCTs(bx_alct, 0, 1) << std::endl
                                                  << allLCTs(bx_alct, 0, 1).getALCT() << std::endl;
                if (allLCTs(bx_alct, 0, 1).getType() == 4)
                  LogTrace("CSCGEMMotherboardME21")
                      << allLCTs(bx_alct, 0, 1).getGEM1() << " " << allLCTs(bx_alct, 0, 1).getGEM2() << std::endl
                      << std::endl;
              }
            }
            if (match_earliest_clct_only)
              break;
          } else {
            LogTrace("CSCGEMMotherboardME21") << "No valid LCT is built from ALCT-GEM matching in ME21" << std::endl;
          }
        }
      }

      if (debug_matching) {
        LogTrace("CSCGEMMotherboardME21")
            << "========================================================================" << std::endl;
        LogTrace("CSCGEMMotherboardME21") << "Summary: " << std::endl;
        if (nGoodMatches > 1)
          LogTrace("CSCGEMMotherboardME21")
              << "Too many good ALCT-CLCT matches in ME21: " << nGoodMatches << ", CSCDetId " << cscId_
              << ", bx_alct = " << bx_alct << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]"
              << std::endl;
        else if (nGoodMatches == 1)
          LogTrace("CSCGEMMotherboardME21")
              << "1 good ALCT-CLCT match in ME21: "
              << " CSCDetId " << cscId_ << ", bx_alct = " << bx_alct << "; match window: [" << bx_clct_start << "; "
              << bx_clct_stop << "]" << std::endl;
        else if (nGoodGEMMatches == 1)
          LogTrace("CSCGEMMotherboardME21")
              << "1 good ALCT-GEM match in ME21: "
              << " CSCDetId " << cscId_ << ", bx_alct = " << bx_alct << "; match window: [" << bx_clct_start << "; "
              << bx_clct_stop << "]" << std::endl;
        else
          LogTrace("CSCGEMMotherboardME21") << "Bad ALCT-CLCT match in ME21: "
                                            << "CSCDetId " << cscId_ << ", bx_alct = " << bx_alct << "; match window: ["
                                            << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
      }
    }
    // at this point we have invalid ALCTs --> try GEM pad matching
    else {
      auto coPads(coPads_[bx_alct]);
      if (!coPads.empty() and buildLCTfromCLCTandGEM_) {
        const int bx_clct_start(bx_alct - match_trig_window_size / 2 - alctClctOffset_);
        const int bx_clct_stop(bx_alct + match_trig_window_size / 2 - alctClctOffset_);

        if (debug_matching) {
          LogTrace("CSCGEMMotherboardME21")
              << "========================================================================" << std::endl
              << "GEM-CLCT matching in ME2/1 chamber: " << cscId_ << "in bx:" << bx_alct << std::endl
              << "------------------------------------------------------------------------" << std::endl;
        }
        // GEM-to-CLCT
        // matching in ME21
        if (buildLCTfromCLCTandGEM_) {
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
                  LogTrace("CSCGEMMotherboardME21")
                      << "Good GEM-CLCT match in ME21: bx_alct = " << bx_alct << "; match window: [" << bx_clct_start
                      << "; " << bx_clct_stop << "]; bx_clct = " << bx_clct << "\n"
                      << "+++ Best CLCT Details: " << clctProc->getBestCLCT(bx_clct) << "\n"
                      << "+++ Second CLCT Details: " << clctProc->getSecondCLCT(bx_clct) << std::endl;

                  LogTrace("CSCGEMMotherboardME21") << "LCT #1 " << allLCTs(bx_alct, mbx, 0) << std::endl
                                                    << allLCTs(bx_alct, mbx, 0).getALCT() << std::endl;
                  if (allLCTs(bx_alct, mbx, 0).getType() == 5)
                    LogTrace("CSCGEMMotherboardME21")
                        << allLCTs(bx_alct, mbx, 0).getGEM1() << " " << allLCTs(bx_alct, mbx, 0).getGEM2() << std::endl;

                  LogTrace("CSCGEMMotherboardME21") << "LCT #2 " << allLCTs(bx_alct, mbx, 1) << std::endl
                                                    << allLCTs(bx_alct, mbx, 1).getALCT() << std::endl;
                  if (allLCTs(bx_alct, mbx, 1).getType() == 5)
                    LogTrace("CSCGEMMotherboardME21")
                        << allLCTs(bx_alct, mbx, 1).getGEM1() << " " << allLCTs(bx_alct, mbx, 1).getGEM2() << std::endl;
                }

                if (match_earliest_clct_only)
                  break;
              }
            }
          }
        }
      }
    }
  }
  // reduction of nLCTs per each BX
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    // counting
    unsigned int n = 0;
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
      for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
        if (allLCTs(bx, mbx, i).isValid()) {
          ++n;
          if (infoV > 0) {
            LogDebug("CSCGEMMotherboardME21")
                << "LCT" << i + 1 << " " << bx << "/" << bx + mbx - match_trig_window_size / 2 << ": "
                << allLCTs(bx, mbx, i) << std::endl;
          }
        }
      }
    }

    // some simple cross-bx sorting algorithms
    if (tmb_cross_bx_algo == 1 and (n > 2)) {
      n = 0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
        for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
          if (allLCTs(bx, pref[mbx], i).isValid()) {
            n++;
            if (n > 2)
              allLCTs(bx, pref[mbx], i).clear();
          }
        }
      }

      n = 0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
        for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
          if (allLCTs(bx, mbx, i).isValid()) {
            n++;
            if (infoV > 0) {
              LogDebug("CSCGEMMotherboardME21")
                  << "LCT" << i + 1 << " " << bx << "/" << bx + mbx - match_trig_window_size / 2 << ": "
                  << allLCTs(bx, mbx, i) << std::endl;
            }
          }
        }
        if (infoV > 0 and n > 0)
          LogDebug("CSCGEMMotherboardME21") << "bx " << bx << " nnLCT:" << n << " " << n << std::endl;
      }
    }  // x-bx sorting
  }

  bool first = true;
  unsigned int n = 0;
  for (const auto& p : readoutLCTs()) {
    if (debug_matching and first) {
      LogTrace("CSCGEMCMotherboardME21") << "========================================================================"
                                         << std::endl;
      LogTrace("CSCGEMCMotherboardME21") << "Counting the final LCTs in CSCGEM Motherboard ME21" << std::endl;
      LogTrace("CSCGEMCMotherboardME21") << "========================================================================"
                                         << std::endl;
      first = false;
      LogTrace("CSCGEMCMotherboardME21") << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
    }
    n++;
    if (debug_matching)
      LogTrace("CSCGEMCMotherboardME21") << "LCT " << n << "  " << p << std::endl;
  }
}

//readout LCTs
std::vector<CSCCorrelatedLCTDigi> CSCGEMMotherboardME21::readoutLCTs() const {
  std::vector<CSCCorrelatedLCTDigi> result;
  allLCTs.getMatched(result);
  if (tmb_cross_bx_algo == 2)
    CSCUpgradeMotherboard::sortLCTs(result, CSCUpgradeMotherboard::sortLCTsByQuality);
  if (tmb_cross_bx_algo == 3)
    CSCUpgradeMotherboard::sortLCTs(result, CSCUpgradeMotherboard::sortLCTsByGEMDphi);
  return result;
}

void CSCGEMMotherboardME21::correlateLCTsGEM(const CSCALCTDigi& bALCT,
                                             const CSCALCTDigi& sALCT,
                                             const CSCCLCTDigi& bCLCT,
                                             const CSCCLCTDigi& sCLCT,
                                             const GEMPadDigiIds& pads,
                                             const GEMCoPadDigiIds& copads,
                                             CSCCorrelatedLCTDigi& lct1,
                                             CSCCorrelatedLCTDigi& lct2) const {
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  CSCCLCTDigi bestCLCT = bCLCT;
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

  if (!copads.empty() or !pads.empty()) {
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
    const bool ok_bb_copad = ok_bb and bb_copad.isValid();
    const bool ok_bs_copad = ok_bs and bs_copad.isValid();
    const bool ok_sb_copad = ok_sb and sb_copad.isValid();
    const bool ok_ss_copad = ok_ss and ss_copad.isValid();

    const bool ok_bb_pad = (not ok_bb_copad) and ok_bb and bb_pad.isValid();
    const bool ok_bs_pad = (not ok_bs_copad) and ok_bs and bs_pad.isValid();
    const bool ok_sb_pad = (not ok_sb_copad) and ok_sb and sb_pad.isValid();
    const bool ok_ss_pad = (not ok_ss_copad) and ok_ss and ss_pad.isValid();

    // possible cases with copad
    if (ok_bb_copad or ok_ss_copad) {
      if (ok_bb_copad)
        lct1 = constructLCTsGEM(bestALCT, bestCLCT, bb_copad, 1);
      if (ok_ss_copad)
        lct2 = constructLCTsGEM(secondALCT, secondCLCT, ss_copad, 2);
    } else if (ok_bs_copad or ok_sb_copad) {
      if (ok_bs_copad)
        lct1 = constructLCTsGEM(bestALCT, secondCLCT, bs_copad, 1);
      if (ok_sb_copad)
        lct2 = constructLCTsGEM(secondALCT, bestCLCT, sb_copad, 2);
    }

    // done processing?
    if (lct1.isValid() and lct2.isValid())
      return;

    // possible cases with pad
    if ((ok_bb_pad or ok_ss_pad) and not(ok_bs_copad or ok_sb_copad)) {
      if (ok_bb_pad)
        lct1 = constructLCTsGEM(bestALCT, bestCLCT, bb_pad, 1);
      if (ok_ss_pad)
        lct2 = constructLCTsGEM(secondALCT, secondCLCT, ss_pad, 2);
    } else if ((ok_bs_pad or ok_sb_pad) and not(ok_bb_copad or ok_ss_copad)) {
      if (ok_bs_pad)
        lct1 = constructLCTsGEM(bestALCT, secondCLCT, bs_pad, 1);
      if (ok_sb_pad)
        lct2 = constructLCTsGEM(secondALCT, bestCLCT, sb_pad, 2);
    }
  } else {
    // run without gems - happens in less than 0.04% of the time
    if (ok_bb)
      lct1 = constructLCTs(bestALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
    if (ok_ss)
      lct2 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
  }
}
