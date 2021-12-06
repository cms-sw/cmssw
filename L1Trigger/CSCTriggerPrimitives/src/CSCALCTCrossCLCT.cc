#include "L1Trigger/CSCTriggerPrimitives/interface/CSCALCTCrossCLCT.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {

  // LUTs to map wiregroup onto min and max half-strip number that it crosses in ME1/1
  // These LUTs are deliberately not implemented as eventsetup objects
  constexpr std::array<int, 48> wg_min_hs_ME1a_ = {
      {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, -1, -1, -1, -1, -1, -1, -1, -1,
       -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1}};
  constexpr std::array<int, 48> wg_max_hs_ME1a_ = {
      {223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 205, 189, 167, 150, -1, -1, -1, -1, -1, -1, -1, -1,
       -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1}};
  constexpr std::array<int, 48> wg_min_hs_ME1a_ganged_ = {
      {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, -1, -1, -1, -1, -1, -1, -1, -1,
       -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1}};
  constexpr std::array<int, 48> wg_max_hs_ME1a_ganged_ = {
      {159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 159, 150, -1, -1, -1, -1, -1, -1, -1, -1,
       -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1}};
  constexpr std::array<int, 48> wg_min_hs_ME1b_ = {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, 73, 47, 22, 0, 0,
                                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0, 0,
                                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0, 0}};
  constexpr std::array<int, 48> wg_max_hs_ME1b_ = {{-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  127, 127,
                                                    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
                                                    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
                                                    127, 127, 127, 127, 127, 127, 127, 127, 105, 93,  78,  63}};
};  // namespace

CSCALCTCrossCLCT::CSCALCTCrossCLCT(
    unsigned endcap, unsigned station, unsigned ring, bool ignoreAlctCrossClct, const edm::ParameterSet& conf)
    : endcap_(endcap), station_(station), ring_(ring) {
  const auto& commonParams = conf.getParameter<edm::ParameterSet>("commonParam");
  gangedME1a_ = commonParams.getParameter<bool>("gangedME1a");
  ignoreAlctCrossClct_ = ignoreAlctCrossClct;
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
  // sanity-check for invalid wiregroups
  if (wiregroup < 0 or wiregroup >= CSCConstants::NUM_WIREGROUPS_ME11)
    return false;

  const int min_hs_ME1a = wg_min_hs_ME1a_[wiregroup];
  const int max_hs_ME1a = wg_max_hs_ME1a_[wiregroup];
  const int min_hs_ME1a_ganged = wg_min_hs_ME1a_ganged_[wiregroup];
  const int max_hs_ME1a_ganged = wg_max_hs_ME1a_ganged_[wiregroup];
  const int min_hs_ME1b = wg_min_hs_ME1b_[wiregroup];
  const int max_hs_ME1b = wg_max_hs_ME1b_[wiregroup];

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
