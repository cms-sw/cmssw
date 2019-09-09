//-----------------------------------------------------------------------------
//
//   Class: CSCMotherboardME11
//
//   Description:
//    Extended CSCMotherboard for ME11 to handle ME1a and ME1b separately
//
//   Author List: Vadim Khotilovich 12 May 2009
//
//
//-----------------------------------------------------------------------------

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboardME11.h"

CSCMotherboardME11::CSCMotherboardME11(unsigned endcap,
                                       unsigned station,
                                       unsigned sector,
                                       unsigned subsector,
                                       unsigned chamber,
                                       const edm::ParameterSet& conf)
    : CSCUpgradeMotherboard(endcap, station, sector, subsector, chamber, conf) {
  if (!isSLHC_)
    edm::LogError("CSCMotherboardME11|ConfigError")
        << "+++ Upgrade CSCMotherboardME11 constructed while isSLHC_ is not set! +++\n";

  cscTmbLUT_.reset(new CSCMotherboardLUTME11());

  // ignore unphysical ALCT-CLCT matches
  ignoreAlctCrossClct = tmbParams_.getParameter<bool>("ignoreAlctCrossClct");
}

CSCMotherboardME11::CSCMotherboardME11() : CSCUpgradeMotherboard() {
  if (!isSLHC_)
    edm::LogError("CSCMotherboardME11|ConfigError")
        << "+++ Upgrade CSCMotherboardME11 constructed while isSLHC_ is not set! +++\n";
}

CSCMotherboardME11::~CSCMotherboardME11() {}

void CSCMotherboardME11::clear() { CSCUpgradeMotherboard::clear(); }

// Set configuration parameters obtained via EventSetup mechanism.
void CSCMotherboardME11::setConfigParameters(const CSCDBL1TPParameters* conf) {
  alctProc->setConfigParameters(conf);
  clctProc->setConfigParameters(conf);
  // No config. parameters in DB for the TMB itself yet.
}

void CSCMotherboardME11::run(const CSCWireDigiCollection* wiredc, const CSCComparatorDigiCollection* compdc) {
  clear();

  // Check for existing processors
  if (!(alctProc && clctProc && isSLHC_)) {
    if (infoV >= 0)
      edm::LogError("CSCMotherboardME11|SetupError") << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alctProc->setCSCGeometry(cscGeometry_);
  clctProc->setCSCGeometry(cscGeometry_);

  alctV = alctProc->run(wiredc);  // run anodeLCT
  clctV = clctProc->run(compdc);  // run cathodeLCT

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV.empty())
    return;

  int used_alct_mask[20];
  int used_clct_mask[20];
  for (int b = 0; b < 20; b++)
    used_alct_mask[b] = used_clct_mask[b] = 0;

  // CLCT-centric CLCT-to-ALCT matching
  if (clct_to_alct) {
    for (int bx_clct = 0; bx_clct < CSCConstants::MAX_CLCT_TBINS; bx_clct++) {
      if (clctProc->getBestCLCT(bx_clct).isValid()) {
        bool is_matched = false;
        const int bx_alct_start = bx_clct - match_trig_window_size / 2 + alctClctOffset_;
        const int bx_alct_stop = bx_clct + match_trig_window_size / 2 + alctClctOffset_;
        for (int bx_alct = bx_alct_start; bx_alct <= bx_alct_stop; bx_alct++) {
          if (bx_alct < 0 || bx_alct >= CSCConstants::MAX_ALCT_TBINS)
            continue;
          if (drop_used_alcts && used_alct_mask[bx_alct])
            continue;
          if (alctProc->getBestALCT(bx_alct).isValid()) {
            if (infoV > 1)
              LogTrace("CSCMotherboardME11")
                  << "Successful CLCT-ALCT match in ME11: bx_clct = " << bx_clct << "; match window: [" << bx_alct_start
                  << "; " << bx_alct_stop << "]; bx_alct = " << bx_alct;
            int mbx = bx_alct_stop - bx_alct;
            correlateLCTsME11(alctProc->getBestALCT(bx_alct),
                              alctProc->getSecondALCT(bx_alct),
                              clctProc->getBestCLCT(bx_clct),
                              clctProc->getSecondCLCT(bx_clct),
                              allLCTs(bx_alct, mbx, 0),
                              allLCTs(bx_alct, mbx, 1));
            if (allLCTs(bx_alct, mbx, 0).isValid()) {
              used_alct_mask[bx_alct] += 1;
              if (match_earliest_alct_only)
                break;
            }
          }
        }
        // Do not report CLCT-only LCT for ME11
        if (!is_matched) {
          if (infoV > 1)
            LogTrace("CSCMotherboard") << "Unsuccessful ALCT-CLCT match (CLCT only): bx_clct = " << bx_clct
                                       << " first CLCT " << clctProc->getBestCLCT(bx_clct) << "; match window: ["
                                       << bx_alct_start << "; " << bx_alct_stop << "]";
        }
      }
    }  // end of CLCT-centric matching

    // ALCT-centric ALCT-to-CLCT matching
  } else {
    for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
      if (alctProc->getBestALCT(bx_alct).isValid()) {
        const int bx_clct_start = bx_alct - match_trig_window_size / 2 - alctClctOffset_;
        const int bx_clct_stop = bx_alct + match_trig_window_size / 2 - alctClctOffset_;

        // matching in ME11
        bool is_matched = false;
        for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
          if (bx_clct < 0 || bx_clct >= CSCConstants::MAX_CLCT_TBINS)
            continue;
          if (drop_used_clcts && used_clct_mask[bx_clct])
            continue;
          if (clctProc->getBestCLCT(bx_clct).isValid()) {
            if (infoV > 1)
              LogTrace("CSCMotherboardME11")
                  << "Successful ALCT-CLCT match in ME11: bx_alct = " << bx_alct << "; match window: [" << bx_clct_start
                  << "; " << bx_clct_stop << "]; bx_clct = " << bx_clct;
            int mbx = bx_clct - bx_clct_start;
            correlateLCTsME11(alctProc->getBestALCT(bx_alct),
                              alctProc->getSecondALCT(bx_alct),
                              clctProc->getBestCLCT(bx_clct),
                              clctProc->getSecondCLCT(bx_clct),
                              allLCTs(bx_alct, mbx, 0),
                              allLCTs(bx_alct, mbx, 1));
            if (allLCTs(bx_alct, mbx, 0).isValid()) {
              is_matched = true;
              used_clct_mask[bx_clct] += 1;
              if (match_earliest_clct_only)
                break;
            }
          }
        }
        if (!is_matched) {
          if (infoV > 1)
            LogTrace("CSCMotherboard") << "Unsuccessful ALCT-CLCT match (ALCT only): bx_alct = " << bx_alct
                                       << " first ALCT " << alctProc->getBestALCT(bx_alct) << "; match window: ["
                                       << bx_clct_start << "; " << bx_clct_stop << "]";
        }
      }
    }  // end of ALCT-centric matching
  }

  // reduction of nLCTs per each BX
  //add similar cross bx algorithm to standard TMB in next step
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    // counting
    unsigned int nlct = 0;
    unsigned int nbx = 0;
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
      bool hasLCT = false;
      for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
        if (allLCTs(bx, mbx, i).isValid()) {
          nlct++;
          hasLCT = true;
          if (infoV > 0) {
            LogDebug("CSCMotherboardME11") << "LCT" << i + 1 << " " << bx << "/"
                                           << bx + mbx - match_trig_window_size / 2 << ": " << allLCTs(bx, mbx, i);
          }
        }
      }
      if (hasLCT)
        nbx++;
    }
    if (infoV > 0 && nlct > 0)
      LogDebug("CSCMotherboardME11") << "bx " << bx << " nLCT: " << nlct << " total mbx with LCTs " << nbx;

    // some simple cross-bx sorting algorithms
    if (tmb_cross_bx_algo == 1 and (nlct > 2 or nbx > 1)) {
      nbx = 0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
        nlct = 0;
        bool hasLCT = false;
        for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
          if (allLCTs(bx, pref[mbx], i).isValid()) {
            nlct++;
            hasLCT = true;
            if (nlct > CSCConstants::MAX_LCTS_PER_CSC or nbx > 0)
              allLCTs(bx, pref[mbx], i).clear();
          }
        }
        if (hasLCT)
          nbx++;
      }

      if (infoV > 0)
        LogDebug("CSCMotherboardME11") << "After x-bx sorting:";
      nlct = 0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
          if (allLCTs(bx, mbx, i).isValid()) {
            nlct++;
            if (infoV > 0) {
              LogDebug("CSCMotherboardME11") << "LCT" << i + 1 << " " << bx << "/"
                                             << bx + mbx - match_trig_window_size / 2 << ": " << allLCTs(bx, mbx, i);
            }
          }
        }
      if (infoV > 0 && nlct > 0)
        LogDebug("CSCMotherboardME11") << "bx " << bx << " nnLCT: " << nlct;
    }  // x-bx sorting
  }    // end of for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++)
}

std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::readoutLCTs1a() const { return readoutLCTs(ME1A); }

std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::readoutLCTs1b() const { return readoutLCTs(ME1B); }

// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::readoutLCTs(int me1ab) const {
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
  if (me1ab == ME1A)
    all_lcts = getLCTs1a();
  if (me1ab == ME1B)
    all_lcts = getLCTs1b();
  std::vector<CSCCorrelatedLCTDigi>::const_iterator plct = all_lcts.begin();
  for (; plct != all_lcts.end(); plct++) {
    if (!plct->isValid())
      continue;

    int bx = (*plct).getBX();
    // Skip LCTs found too early relative to L1Accept.
    if (bx <= early_tbins)
      continue;

    // Skip LCTs found too late relative to L1Accept.
    if (bx > late_tbins)
      continue;

    // If (readout_earliest_2) take only LCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2 && (bx_readout == -1 || bx == bx_readout)) {
      tmpV.push_back(*plct);
      if (bx_readout == -1)
        bx_readout = bx;
    } else
      tmpV.push_back(*plct);
  }
  return tmpV;
}

// Returns vector of found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::getLCTs1b() const {
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
      for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
        const CSCCorrelatedLCTDigi& lct = allLCTs.data[bx][mbx][i];
        if (lct.isValid() and lct.getStrip() < CSCConstants::MAX_HALF_STRIP_ME1B) {
          tmpV.push_back(lct);
        }
      }
    }
  }
  return tmpV;
}

// Returns vector of found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11::getLCTs1a() const {
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  // disabled ME1a
  if (mpc_block_me1a || disableME1a_)
    return tmpV;

  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
      for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
        const CSCCorrelatedLCTDigi& lct = allLCTs.data[bx][mbx][i];
        if (lct.isValid() and lct.getStrip() >= CSCConstants::MAX_HALF_STRIP_ME1B) {
          tmpV.push_back(lct);
        }
      }
    }
  }  // Report all LCTs found.
  return tmpV;
}

bool CSCMotherboardME11::doesALCTCrossCLCT(const CSCALCTDigi& a, const CSCCLCTDigi& c) const {
  return cscTmbLUT_->doesALCTCrossCLCT(a, c, theEndcap, gangedME1a_);
}

void CSCMotherboardME11::correlateLCTsME11(const CSCALCTDigi& bALCT,
                                           const CSCALCTDigi& sALCT,
                                           const CSCCLCTDigi& bCLCT,
                                           const CSCCLCTDigi& sCLCT,
                                           CSCCorrelatedLCTDigi& lct1,
                                           CSCCorrelatedLCTDigi& lct2) const {
  // assume that always anodeBestValid && cathodeBestValid
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  CSCCLCTDigi bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

  if (ignoreAlctCrossClct) {
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
    return;
  } else {
    if (secondALCT == bestALCT)
      secondALCT.clear();
    if (secondCLCT == bestCLCT)
      secondCLCT.clear();

    const int ok11 = doesALCTCrossCLCT(bestALCT, bestCLCT);
    const int ok12 = doesALCTCrossCLCT(bestALCT, secondCLCT);
    const int ok21 = doesALCTCrossCLCT(secondALCT, bestCLCT);
    const int ok22 = doesALCTCrossCLCT(secondALCT, secondCLCT);
    const int code = (ok11 << 3) | (ok12 << 2) | (ok21 << 1) | (ok22);

    int dbg = 0;
    if (dbg)
      LogTrace("CSCMotherboardME11") << "debug correlateLCTs in ME11 " << cscId_ << std::endl
                                     << "ALCT1: " << bestALCT << std::endl
                                     << "ALCT2: " << secondALCT << std::endl
                                     << "CLCT1: " << bestCLCT << std::endl
                                     << "CLCT2: " << secondCLCT << std::endl
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
      LogTrace("CSCMotherboardME11") << "lut 0 1 = " << lut[code][0] << " " << lut[code][1] << std::endl;

    switch (lut[code][0]) {
      case 11:
        lct1 = constructLCTs(bestALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
        break;
      case 12:
        lct1 = constructLCTs(bestALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
        break;
      case 21:
        lct1 = constructLCTs(secondALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
        break;
      case 22:
        lct1 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 1);
        break;
      default:
        return;
    }

    if (dbg)
      LogTrace("CSCMotherboardME11") << "lct1: " << lct1 << std::endl;

    switch (lut[code][1]) {
      case 12:
        lct2 = constructLCTs(bestALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
        if (dbg)
          LogTrace("CSCMotherboardME11") << "lct2: " << lct2 << std::endl;
        return;
      case 21:
        lct2 = constructLCTs(secondALCT, bestCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
        if (dbg)
          LogTrace("CSCMotherboardME11") << "lct2: " << lct2 << std::endl;
        return;
      case 22:
        lct2 = constructLCTs(secondALCT, secondCLCT, CSCCorrelatedLCTDigi::ALCTCLCT, 2);
        if (dbg)
          LogTrace("CSCMotherboardME11") << "lct2: " << lct2 << std::endl;
        return;
      default:
        return;
    }
    if (dbg)
      LogTrace("CSCMotherboardME11") << "out of correlateLCTsME11" << std::endl;

    return;
  }
}
