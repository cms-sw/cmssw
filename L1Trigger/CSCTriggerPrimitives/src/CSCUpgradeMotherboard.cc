#include <memory>

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeMotherboard.h"

CSCUpgradeMotherboard::CSCUpgradeMotherboard(unsigned endcap,
                                             unsigned station,
                                             unsigned sector,
                                             unsigned subsector,
                                             unsigned chamber,
                                             const edm::ParameterSet& conf)
    :  // special configuration parameters for ME11 treatment
      CSCMotherboard(endcap, station, sector, subsector, chamber, conf),
      allLCTs(match_trig_window_size) {
  if (!runPhase2_)
    edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runPhase2_ is not set! +++\n";

  if (theRing == 1) {
    if (theStation == 1 and !runME11Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME11Up_ is not set! +++\n";
    if (theStation == 2 and !runME21Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME21Up_ is not set! +++\n";
    if (theStation == 3 and !runME31Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME31Up_ is not set! +++\n";
    if (theStation == 4 and !runME41Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME41Up_ is not set! +++\n";
  }

  theParity = theChamber % 2 == 0 ? Parity::Even : Parity::Odd;

  // generate the LUTs
  generator_ = std::make_unique<CSCUpgradeMotherboardLUTGenerator>();

  // enable the upgrade processors
  if (runPhase2_ and theRing == 1) {
    clctProc = std::make_unique<CSCUpgradeCathodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);
    if (enableAlctPhase2_) {
      alctProc = std::make_unique<CSCUpgradeAnodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);
    }
  }

  match_earliest_alct_only = tmbParams_.getParameter<bool>("matchEarliestAlctOnly");
  match_earliest_clct_only = tmbParams_.getParameter<bool>("matchEarliestClctOnly");
  clct_to_alct = tmbParams_.getParameter<bool>("clctToAlct");
  drop_used_clcts = tmbParams_.getParameter<bool>("tmbDropUsedClcts");
  tmb_cross_bx_algo = tmbParams_.getParameter<unsigned int>("tmbCrossBxAlgorithm");
  max_lcts = tmbParams_.getParameter<unsigned int>("maxLCTs");
  debug_matching = tmbParams_.getParameter<bool>("debugMatching");
  debug_luts = tmbParams_.getParameter<bool>("debugLUTs");

  setPrefIndex();
}

CSCUpgradeMotherboard::CSCUpgradeMotherboard() : CSCMotherboard(), allLCTs(match_trig_window_size) {
  if (!runPhase2_)
    edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runPhase2_ is not set! +++\n";

  if (theRing == 1) {
    if (theStation == 1 and !runME11Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME11Up_ is not set! +++\n";
    if (theStation == 2 and !runME21Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME21Up_ is not set! +++\n";
    if (theStation == 3 and !runME31Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME31Up_ is not set! +++\n";
    if (theStation == 4 and !runME41Up_)
      edm::LogError("CSCUpgradeMotherboard|SetupError") << "+++ TMB constructed while runME41Up_ is not set! +++\n";
  }

  setPrefIndex();
}

void CSCUpgradeMotherboard::run(const CSCWireDigiCollection* wiredc, const CSCComparatorDigiCollection* compdc) {
  clear();

  if (!(alctProc and clctProc)) {
    edm::LogError("CSCUpgradeMotherboard|SetupError")
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

  // encode high multiplicity bits
  unsigned alctBits = alctProc->getHighMultiplictyBits();
  encodeHighMultiplicityBits(alctBits);

  int used_clct_mask[20];
  for (int c = 0; c < 20; ++c)
    used_clct_mask[c] = 0;

  // ALCT centric matching
  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
    if (alctProc->getBestALCT(bx_alct).isValid()) {
      const int bx_clct_start(bx_alct - match_trig_window_size / 2 - CSCConstants::ALCT_CLCT_OFFSET);
      const int bx_clct_stop(bx_alct + match_trig_window_size / 2 - CSCConstants::ALCT_CLCT_OFFSET);

      if (debug_matching) {
        LogTrace("CSCUpgradeMotherboard")
            << "========================================================================" << std::endl;
        LogTrace("CSCUpgradeMotherboard") << "ALCT-CLCT matching in ME34/1 chamber: " << cscId_ << std::endl;
        LogTrace("CSCUpgradeMotherboard")
            << "------------------------------------------------------------------------" << std::endl;
        LogTrace("CSCUpgradeMotherboard") << "+++ Best ALCT Details: ";
        alctProc->getBestALCT(bx_alct).print();
        LogTrace("CSCUpgradeMotherboard") << "+++ Second ALCT Details: ";
        alctProc->getSecondALCT(bx_alct).print();

        LogTrace("CSCUpgradeMotherboard")
            << "------------------------------------------------------------------------" << std::endl;
        LogTrace("CSCUpgradeMotherboard") << "Attempt ALCT-CLCT matching in ME34/13 in bx range: [" << bx_clct_start
                                          << "," << bx_clct_stop << "]" << std::endl;
      }

      // ALCT-to-CLCT
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
        if (bx_clct < 0 or bx_clct >= CSCConstants::MAX_CLCT_TBINS)
          continue;
        if (drop_used_clcts and used_clct_mask[bx_clct])
          continue;
        if (clctProc->getBestCLCT(bx_clct).isValid()) {
          if (debug_matching)
            LogTrace("CSCUpgradeMotherboard") << "++Valid ME21 CLCT: " << clctProc->getBestCLCT(bx_clct) << std::endl;

          int mbx = bx_clct - bx_clct_start;
          CSCUpgradeMotherboard::correlateLCTs(alctProc->getBestALCT(bx_alct),
                                               alctProc->getSecondALCT(bx_alct),
                                               clctProc->getBestCLCT(bx_clct),
                                               clctProc->getSecondCLCT(bx_clct),
                                               allLCTs(bx_alct, mbx, 0),
                                               allLCTs(bx_alct, mbx, 1));
          if (infoV > 1)
            LogTrace("CSCUpgradeMotherboard")
                << "Successful ALCT-CLCT match in ME21: bx_alct = " << bx_alct << "; match window: [" << bx_clct_start
                << "; " << bx_clct_stop << "]; bx_clct = " << bx_clct << std::endl;
          LogTrace("CSCUpgradeMotherboard") << "+++ Best CLCT Details: ";
          clctProc->getBestCLCT(bx_clct).print();
          LogTrace("CSCUpgradeMotherboard") << "+++ Second CLCT Details: ";
          clctProc->getSecondCLCT(bx_clct).print();
          if (allLCTs(bx_alct, mbx, 0).isValid()) {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_only)
              break;
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
            LogDebug("CSCUpgradeMotherboard")
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
              LogDebug("CSCUpgradeMotherboard")
                  << "LCT" << i + 1 << " " << bx << "/" << bx + mbx - match_trig_window_size / 2 << ": "
                  << allLCTs(bx, mbx, i) << std::endl;
            }
          }
        }
      }
      if (infoV > 0 and n > 0)
        LogDebug("CSCUpgradeMotherboard") << "bx " << bx << " nnLCT:" << n << " " << n << std::endl;
    }  // x-bx sorting
  }

  bool first = true;
  unsigned int n = 0;
  for (const auto& p : readoutLCTs()) {
    if (debug_matching and first) {
      LogTrace("CSCUpgradeMotherboard") << "========================================================================"
                                        << std::endl;
      LogTrace("CSCUpgradeMotherboard") << "Counting the final LCTs" << std::endl;
      LogTrace("CSCUpgradeMotherboard") << "========================================================================"
                                        << std::endl;
      first = false;
      LogTrace("CSCUpgradeMotherboard") << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
    }
    n++;
    if (debug_matching)
      LogTrace("CSCUpgradeMotherboard") << "LCT " << n << "  " << p << std::endl;
  }
}

void CSCUpgradeMotherboard::correlateLCTs(const CSCALCTDigi& bALCT,
                                          const CSCALCTDigi& sALCT,
                                          const CSCCLCTDigi& bCLCT,
                                          const CSCCLCTDigi& sCLCT,
                                          CSCCorrelatedLCTDigi& lct1,
                                          CSCCorrelatedLCTDigi& lct2) const {
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  CSCCLCTDigi bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

  const bool anodeBestValid = bestALCT.isValid();
  const bool anodeSecondValid = secondALCT.isValid();
  const bool cathodeBestValid = bestCLCT.isValid();
  const bool cathodeSecondValid = secondCLCT.isValid();

  if (anodeBestValid and !anodeSecondValid)
    secondALCT = bestALCT;
  if (!anodeBestValid and anodeSecondValid)
    bestALCT = secondALCT;
  if (cathodeBestValid and !cathodeSecondValid)
    secondCLCT = bestCLCT;
  if (!cathodeBestValid and cathodeSecondValid)
    bestCLCT = secondCLCT;

  // ALCT-CLCT matching conditions are defined by "trig_enable" configuration
  // parameters.
  if ((alct_trig_enable and bestALCT.isValid()) or (clct_trig_enable and bestCLCT.isValid()) or
      (match_trig_enable and bestALCT.isValid() and bestCLCT.isValid())) {
    lct1 = constructLCTs(bestALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
  }

  if (((secondALCT != bestALCT) or (secondCLCT != bestCLCT)) and
      ((alct_trig_enable and secondALCT.isValid()) or (clct_trig_enable and secondCLCT.isValid()) or
       (match_trig_enable and secondALCT.isValid() and secondCLCT.isValid()))) {
    lct2 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
  }
}

//readout LCTs
std::vector<CSCCorrelatedLCTDigi> CSCUpgradeMotherboard::readoutLCTs() const {
  std::vector<CSCCorrelatedLCTDigi> result;
  allLCTs.getMatched(result);
  if (tmb_cross_bx_algo == 2)
    CSCUpgradeMotherboard::sortLCTs(result, CSCUpgradeMotherboard::sortLCTsByQuality);

  // do a final check on the LCTs in readout
  qualityControl_->checkMultiplicityBX(result);
  for (const auto& lct : result) {
    qualityControl_->checkValid(lct);
  }

  return result;
}

CSCUpgradeMotherboard::~CSCUpgradeMotherboard() {}

enum CSCPart CSCUpgradeMotherboard::getCSCPart(int keystrip) const {
  if (theStation == 1 and (theRing == 1 or theRing == 4)) {
    if (keystrip > CSCConstants::MAX_HALF_STRIP_ME1B) {
      if (gangedME1a_)
        return CSCPart::ME1Ag;
      else
        return CSCPart::ME1A;
    } else if (keystrip <= CSCConstants::MAX_HALF_STRIP_ME1B and keystrip >= 0)
      return CSCPart::ME1B;
    else
      return CSCPart::ME11;
  } else if (theStation == 2 and theRing == 1)
    return CSCPart::ME21;
  else if (theStation == 3 and theRing == 1)
    return CSCPart::ME31;
  else if (theStation == 4 and theRing == 1)
    return CSCPart::ME41;
  else {
    edm::LogError("CSCUpgradeMotherboard|Error") << " ++ getCSCPart() failed to find the CSC chamber for in case ";
    return CSCPart::ME11;  // return ME11 by default
  }
}

void CSCUpgradeMotherboard::debugLUTs() {
  if (debug_luts)
    generator_->generateLUTs(theEndcap, theStation, theSector, theSubsector, theTrigChamber);
}

bool CSCUpgradeMotherboard::sortLCTsByQuality(const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2) {
  return lct1.getQuality() > lct2.getQuality();
}

bool CSCUpgradeMotherboard::sortLCTsByGEMDphi(const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2) {
  return true;
}

void CSCUpgradeMotherboard::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts,
                                     bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const {
  std::sort(lcts.begin(), lcts.end(), *sorter);
  if (lcts.size() > max_lcts)
    lcts.erase(lcts.begin() + max_lcts, lcts.end());
}

void CSCUpgradeMotherboard::setupGeometry() { generator_->setCSCGeometry(cscGeometry_); }

void CSCUpgradeMotherboard::setPrefIndex() {
  pref[0] = match_trig_window_size / 2;
  for (unsigned int m = 2; m < match_trig_window_size; m += 2) {
    pref[m - 1] = pref[0] - m / 2;
    pref[m] = pref[0] + m / 2;
  }
}

void CSCUpgradeMotherboard::clear() {
  CSCMotherboard::clear();
  allLCTs.clear();
}
