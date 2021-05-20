#include "L1Trigger/CSCTriggerPrimitives/interface/CSCALCTCrossCLCT.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCLUTReader.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"

CSCALCTCrossCLCT::CSCALCTCrossCLCT(
    unsigned endcap, unsigned station, unsigned ring, bool isganged, const edm::ParameterSet& luts)
    : endcap_(endcap), station_(station), ring_(ring), isganged_(isganged) {
  wgCrossHsME1aFiles_ = luts.getParameter<std::vector<std::string>>("wgCrossHsME1aFiles");
  wgCrossHsME1aGangedFiles_ = luts.getParameter<std::vector<std::string>>("wgCrossHsME1aGangedFiles");
  wgCrossHsME1bFiles_ = luts.getParameter<std::vector<std::string>>("wgCrossHsME1bFiles");

  wg_cross_min_hs_ME1a_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aFiles_[0]);
  wg_cross_max_hs_ME1a_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aFiles_[1]);
  wg_cross_min_hs_ME1a_ganged_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aGangedFiles_[0]);
  wg_cross_max_hs_ME1a_ganged_ = std::make_unique<CSCLUTReader>(wgCrossHsME1aGangedFiles_[1]);
  wg_cross_min_hs_ME1b_ = std::make_unique<CSCLUTReader>(wgCrossHsME1bFiles_[0]);
  wg_cross_max_hs_ME1b_ = std::make_unique<CSCLUTReader>(wgCrossHsME1bFiles_[1]);
}

bool CSCALCTCrossCLCT::doesALCTCrossCLCT(const CSCALCTDigi& a, const CSCCLCTDigi& c, bool ignoreAlctCrossClct) const {
  // both need to valid
  if (!c.isValid() || !a.isValid()) {
    return false;
  }

  // when non-overlapping half-strips and wiregroups don't matter
  if (ignoreAlctCrossClct)
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
    if (!isganged_) {
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
