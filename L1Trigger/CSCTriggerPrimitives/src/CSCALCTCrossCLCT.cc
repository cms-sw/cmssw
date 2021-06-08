#include "L1Trigger/CSCTriggerPrimitives/interface/CSCALCTCrossCLCT.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCLUTReader.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

CSCALCTCrossCLCT::CSCALCTCrossCLCT(
    unsigned endcap, unsigned station, unsigned ring, bool ignoreAlctCrossClct, const edm::ParameterSet& conf)
    : endcap_(endcap), station_(station), ring_(ring) {
  const auto& commonParams = conf.getParameter<edm::ParameterSet>("commonParam");
  gangedME1a_ = commonParams.getParameter<bool>("gangedME1a");
  ignoreAlctCrossClct_ = ignoreAlctCrossClct;

  const edm::ParameterSet me11luts(conf.getParameter<edm::ParameterSet>("wgCrossHsME11Params"));
  wgCrossHsME1aFiles_ = me11luts.getParameter<std::vector<std::string>>("wgCrossHsME1aFiles");
  wgCrossHsME1aGangedFiles_ = me11luts.getParameter<std::vector<std::string>>("wgCrossHsME1aGangedFiles");
  wgCrossHsME1bFiles_ = me11luts.getParameter<std::vector<std::string>>("wgCrossHsME1bFiles");
  wg_cross_min_hs_ME1a_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aFiles_[0]);
  wg_cross_max_hs_ME1a_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aFiles_[1]);
  wg_cross_min_hs_ME1a_ganged_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aGangedFiles_[0]);
  wg_cross_max_hs_ME1a_ganged_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aGangedFiles_[1]);
  wg_cross_min_hs_ME1b_ = std::make_unique<CSCLUTReader>(wgCrossHsME1bFiles_[0]);
  wg_cross_max_hs_ME1b_ = std::make_unique<CSCLUTReader>(wgCrossHsME1bFiles_[1]);

  const edm::ParameterSet lctCodeluts(conf.getParameter<edm::ParameterSet>("lctCodeParams"));
  lctCombinationCodeFiles_ = lctCodeluts.getParameter<std::vector<std::string>>("lctCodeFiles");
  code_to_best_lct_ = std::make_unique<CSCLUTReader>(lctCombinationCodeFiles_[0]);
  code_to_second_lct_ = std::make_unique<CSCLUTReader>(lctCombinationCodeFiles_[1]);
}

void CSCALCTCrossCLCT::calculateLCTCodes(const CSCALCTDigi& bestALCT,
                                         const CSCCLCTDigi& bestCLCT,
                                         const CSCALCTDigi& secondALCT,
                                         const CSCCLCTDigi& secondCLCT,
                                         unsigned& bestLCTCode,
                                         unsigned& secondLCTCode) const {
  // Each of these calls should return "1" when the ALCT and CLCT are valid.
  const bool ok11 = doesALCTCrossCLCT(bestALCT, bestCLCT);
  const bool ok12 = doesALCTCrossCLCT(bestALCT, secondCLCT);
  const bool ok21 = doesALCTCrossCLCT(secondALCT, bestCLCT);
  const bool ok22 = doesALCTCrossCLCT(secondALCT, secondCLCT);

  /*
    With these okxx, we now calculate a 4-bit code that determines
    the best and second LCT combinations.
  */

  const unsigned code = (ok11 << 3) | (ok12 << 2) | (ok21 << 1) | (ok22);

  bestLCTCode = code_to_best_lct_->lookup(code);
  secondLCTCode = code_to_second_lct_->lookup(code);

  edm::LogInfo("CSCALCTCrossCLCT") << "Calculate LCT combination code" << std::endl
                                   << "ALCT1: " << bestALCT << std::endl
                                   << "ALCT2: " << secondALCT << std::endl
                                   << "CLCT1: " << bestCLCT << std::endl
                                   << "CLCT2: " << secondCLCT << std::endl
                                   << "LCT combination code: " << code << std::endl
                                   << "LCT1: " << bestLCTCode << std::endl
                                   << "LCT1: " << secondLCTCode << std::endl;
}

bool CSCALCTCrossCLCT::doesALCTCrossCLCT(const CSCALCTDigi& a, const CSCCLCTDigi& c) const {
  // both need to valid
  if (!c.isValid() || !a.isValid()) {
    return false;
  }

  // when non-overlapping half-strips and wiregroups don't matter
  if (ignoreAlctCrossClct_)
    return true;

  // overlap only needs to be considered for ME1/1
  if (station_ == 1 and ring_ == 1) {
    return doesWiregroupCrossHalfStrip(a.getKeyWG(), c.getKeyStrip());
  } else
    return true;
}

bool CSCALCTCrossCLCT::doesWiregroupCrossHalfStrip(int wiregroup, int halfstrip) const {
  const int min_hs_ME1a = wg_cross_min_hs_ME1a_->lookup(wiregroup);
  const int max_hs_ME1a = wg_cross_max_hs_ME1a_->lookup(wiregroup);
  const int min_hs_ME1a_ganged = wg_cross_min_hs_ME1a_ganged_->lookup(wiregroup);
  const int max_hs_ME1a_ganged = wg_cross_max_hs_ME1a_ganged_->lookup(wiregroup);
  const int min_hs_ME1b = wg_cross_min_hs_ME1b_->lookup(wiregroup);
  const int max_hs_ME1b = wg_cross_max_hs_ME1b_->lookup(wiregroup);

  // ME1/a half-strip starts at 128
  if (halfstrip > CSCConstants::MAX_HALF_STRIP_ME1B) {
    if (!gangedME1a_) {
      // wrap around ME11 HS number for -z endcap
      if (endcap_ == 2) {
        // first subtract 128
        halfstrip -= 1 + CSCConstants::MAX_HALF_STRIP_ME1B;
        // flip the HS
        halfstrip = CSCConstants::MAX_HALF_STRIP_ME1A_UNGANGED - halfstrip;
        // then add 128 again
        halfstrip += 1 + CSCConstants::MAX_HALF_STRIP_ME1B;
      }
      return halfstrip >= min_hs_ME1a && halfstrip <= max_hs_ME1a;
    }

    else {
      // wrap around ME11 HS number for -z endcap
      if (endcap_ == 2) {
        // first subtract 128
        halfstrip -= 1 + CSCConstants::MAX_HALF_STRIP_ME1B;
        // flip the HS
        halfstrip = CSCConstants::MAX_HALF_STRIP_ME1A_GANGED - halfstrip;
        // then add 128 again
        halfstrip += 1 + CSCConstants::MAX_HALF_STRIP_ME1B;
      }
      return halfstrip >= min_hs_ME1a_ganged && halfstrip <= max_hs_ME1a_ganged;
    }
  }
  // ME1/b half-strip ends at 127
  else if (halfstrip <= CSCConstants::MAX_HALF_STRIP_ME1B) {
    if (endcap_ == 2) {
      halfstrip = CSCConstants::MAX_HALF_STRIP_ME1B - halfstrip;
    }
    return halfstrip >= min_hs_ME1b && halfstrip <= max_hs_ME1b;
  }
  // all other cases, return true
  else
    return true;
}
