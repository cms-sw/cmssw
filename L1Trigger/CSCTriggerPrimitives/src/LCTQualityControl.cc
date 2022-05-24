#include "L1Trigger/CSCTriggerPrimitives/interface/LCTQualityControl.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/LCTQualityAssignment.h"
#include <unordered_map>

// constructor
LCTQualityControl::LCTQualityControl(unsigned endcap,
                                     unsigned station,
                                     unsigned sector,
                                     unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf)
    : CSCBaseboard(endcap, station, sector, subsector, chamber, conf) {
  nplanes_clct_hit_pattern = clctParams_.getParameter<unsigned int>("clctNplanesHitPattern");
}

// Check if the ALCT is valid
void LCTQualityControl::checkValidReadout(const CSCALCTDigi& alct) const {
  checkValid(alct, CSCConstants::MAX_ALCTS_READOUT);
}

void LCTQualityControl::checkRange(
    int value, int min_value, int max_value, const std::string& comment, unsigned& errors) const {
  if (value < min_value or value > max_value) {
    edm::LogError("LCTQualityControl") << comment << value << "; allowed [" << min_value << ", " << max_value << "]";
    errors++;
  }
}

template <class T>
void LCTQualityControl::reportErrors(const T& lct, const unsigned errors) const {
  if (errors > 0) {
    edm::LogError("LCTQualityControl") << "Invalid stub in " << cscId_ << " (" << errors << " errors):\n" << lct;
  }
}

// Check if the ALCT is valid
void LCTQualityControl::checkValid(const CSCALCTDigi& alct, unsigned max_stubs) const {
  const unsigned max_wiregroup = get_csc_max_wiregroup(theStation, theRing);
  const auto& [min_quality, max_quality] = get_csc_alct_min_max_quality();

  unsigned errors = 0;

  // stub must be valid
  checkRange(alct.isValid(), 1, 1, "CSCALCTDigi with invalid bit set: ", errors);

  // ALCT number is 1 or 2
  checkRange(alct.getTrknmb(), 1, max_stubs, "CSCALCTDigi with invalid track number: ", errors);

  // ALCT quality must be valid
  // number of layers - 3
  checkRange(alct.getQuality(), min_quality, max_quality, "CSCALCTDigi with invalid quality: ", errors);

  // ALCT key wire-group must be within bounds
  checkRange(alct.getKeyWG(), 0, max_wiregroup - 1, "CSCALCTDigi with invalid wire-group: ", errors);

  // ALCT with out-of-time BX
  checkRange(alct.getBX(), 0, CSCConstants::MAX_ALCT_TBINS - 1, "CSCALCTDigi with invalid BX: ", errors);

  // ALCT is neither accelerator or collision
  checkRange(alct.getCollisionB(), 0, 1, "CSCALCTDigi with invalid accel/coll biit: ", errors);

  reportErrors(alct, errors);
}

// Check if the CLCT is valid
void LCTQualityControl::checkValid(const CSCCLCTDigi& clct, unsigned max_stubs) const {
  const unsigned max_strip = get_csc_max_halfstrip(theStation, theRing);
  const auto& [min_pattern_run2, max_pattern_run2] = get_csc_min_max_pattern();
  const auto& [min_pattern_run3, max_pattern_run3] = get_csc_min_max_pattern_run3();
  const auto& [min_slope, max_slope] = get_csc_clct_min_max_slope();
  const auto& [min_cfeb, max_cfeb] = get_csc_min_max_cfeb();
  const auto& [min_quality, max_quality] = get_csc_clct_min_max_quality();

  unsigned errors = 0;

  // CLCT must be valid
  checkRange(clct.isValid(), 1, 1, "CSCCLCTDigi with invalid bit set: ", errors);

  // CLCT number is 1 or max
  checkRange(clct.getTrknmb(), 1, max_stubs, "CSCCLCTDigi with invalid track number: ", errors);

  // CLCT quality must be valid
  // CLCTs require at least 4 layers hit
  // Run-3: ME1/1 CLCTs require only 3 layers
  // Run-4: ME2/1 CLCTs require only 3 layers
  checkRange(clct.getQuality(), min_quality, max_quality, "CSCCLCTDigi with invalid quality: ", errors);

  // CLCT half-strip must be within bounds
  checkRange(
      clct.getStrip(), 0, CSCConstants::NUM_HALF_STRIPS_PER_CFEB - 1, "CSCCLCTDigi with invalid half-strip: ", errors);

  // CLCT key half-strip must be within bounds
  checkRange(clct.getKeyStrip(), 0, max_strip - 1, "CSCCLCTDigi with invalid key half-strip: ", errors);

  // CLCT with out-of-time BX
  checkRange(clct.getBX(), 0, CSCConstants::MAX_CLCT_TBINS - 1, "CSCCLCTDigi with invalid BX: ", errors);

  // CLCT with neither left nor right bending
  checkRange(clct.getBend(), 0, 1, "CSCCLCTDigi with invalid bending: ", errors);

  // CLCT with an invalid pattern ID
  checkRange(
      clct.getPattern(), min_pattern_run2, max_pattern_run2, "CSCCLCTDigi with invalid Run-2 pattern ID: ", errors);

  // CLCT with an invalid pattern ID
  checkRange(
      clct.getRun3Pattern(), min_pattern_run3, max_pattern_run3, "CSCCLCTDigi with invalid Run-3 pattern ID: ", errors);

  // CLCT with an invalid slope
  checkRange(clct.getSlope(), min_slope, max_slope, "CSCCLCTDigi with invalid slope: ", errors);

  // CLCT with an invalid CFEB ID
  checkRange(clct.getCFEB(), min_cfeb, max_cfeb, "CSCCLCTDigi with invalid CFEB ID: ", errors);

  if (runCCLUT_) {
    // CLCT comparator code is invalid
    checkRange(clct.getCompCode(), 0, std::pow(2, 12) - 1, "CSCCLCTDigi with invalid comparator code: ", errors);

    const unsigned max_quartstrip = get_csc_max_quartstrip(theStation, theRing);
    const unsigned max_eighthstrip = get_csc_max_eighthstrip(theStation, theRing);

    // CLCT key quart-strip must be within bounds
    checkRange(clct.getKeyStrip(4), 0, max_quartstrip - 1, "CSCCLCTDigi with invalid key quart-strip: ", errors);

    // CLCT key eighth-strip must be within bounds
    checkRange(clct.getKeyStrip(8), 0, max_eighthstrip - 1, "CSCCLCTDigi with invalid key quart-strip: ", errors);
  }

  reportErrors(clct, errors);
}

void LCTQualityControl::checkValid(const CSCCorrelatedLCTDigi& lct) const { checkValid(lct, theStation, theRing); }

void LCTQualityControl::checkValid(const CSCCorrelatedLCTDigi& lct, unsigned station, unsigned ring) const {
  const unsigned max_strip = get_csc_max_halfstrip(station, ring);
  const unsigned max_quartstrip = get_csc_max_quartstrip(station, ring);
  const unsigned max_eighthstrip = get_csc_max_eighthstrip(station, ring);
  const unsigned max_wiregroup = get_csc_max_wiregroup(station, ring);
  const auto& [min_pattern_run2, max_pattern_run2] = get_csc_min_max_pattern();
  const auto& [min_pattern_run3, max_pattern_run3] = get_csc_min_max_pattern_run3();
  const auto& [min_quality, max_quality] = get_csc_lct_min_max_quality(station, ring);

  unsigned errors = 0;

  // LCT must be valid
  checkRange(lct.isValid(), 1, 1, "CSCCorrelatedLCTDigi with invalid bit set: ", errors);

  // LCT number is 1 or 2
  checkRange(lct.getTrknmb(), 1, 2, "CSCCorrelatedLCTDigi with invalid track number: ", errors);

  // LCT quality must be valid
  checkRange(lct.getQuality(), min_quality, max_quality, "CSCCorrelatedLCTDigi with invalid quality: ", errors);

  // LCT key half-strip must be within bounds
  checkRange(lct.getStrip(), 0, max_strip - 1, "CSCCorrelatedLCTDigi with invalid key half-strip: ", errors);

  // LCT key quart-strip must be within bounds
  checkRange(lct.getStrip(4), 0, max_quartstrip - 1, "CSCCorrelatedLCTDigi with invalid key quart-strip: ", errors);

  // LCT key eighth-strip must be within bounds
  checkRange(lct.getStrip(8), 0, max_eighthstrip - 1, "CSCCorrelatedLCTDigi with invalid key eighth-strip: ", errors);

  // LCT key wire-group must be within bounds
  checkRange(lct.getKeyWG(), 0, max_wiregroup - 1, "CSCCorrelatedLCTDigi with invalid wire-group: ", errors);

  // LCT with out-of-time BX
  checkRange(lct.getBX(), 0, CSCConstants::MAX_LCT_TBINS - 1, "CSCCorrelatedLCTDigi with invalid BX: ", errors);

  // LCT with neither left nor right bending
  checkRange(lct.getBend(), 0, 1, "CSCCorrelatedLCTDigi with invalid bending: ", errors);

  // LCT with invalid MPC link
  checkRange(lct.getMPCLink(), 0, CSCConstants::MAX_LCTS_PER_MPC, "CSCCorrelatedLCTDigi with MPC link: ", errors);

  // LCT with invalid CSCID
  checkRange(lct.getCSCID(),
             CSCTriggerNumbering::minTriggerCscId(),
             CSCTriggerNumbering::maxTriggerCscId(),
             "CSCCorrelatedLCTDigi with invalid CSCID: ",
             errors);

  // LCT with an invalid pattern ID
  checkRange(lct.getPattern(),
             min_pattern_run2,
             max_pattern_run2,
             "CSCCorrelatedLCTDigi with invalid Run-2 pattern ID: ",
             errors);

  checkRange(lct.getRun3Pattern(),
             min_pattern_run3,
             max_pattern_run3,
             "CSCCorrelatedLCTDigi with invalid Run-3 pattern ID: ",
             errors);

  // simulated LCT type must be valid
  if (lct.getType() == CSCCorrelatedLCTDigi::CLCTALCT or lct.getType() == CSCCorrelatedLCTDigi::CLCTONLY or
      lct.getType() == CSCCorrelatedLCTDigi::ALCTONLY) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid type (SIM): " << lct.getType()
                                       << "; allowed [" << CSCCorrelatedLCTDigi::ALCTCLCT << ", "
                                       << CSCCorrelatedLCTDigi::CLCT2GEM << "]";
    errors++;
  }

  // non-GEM-CSC stations ALWAYS send out ALCTCLCT type LCTs
  if (!isME11_ and !isME21_) {
    if (lct.getType() != CSCCorrelatedLCTDigi::ALCTCLCT) {
      edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid type (SIM) in this station: "
                                         << lct.getType() << "; allowed [" << CSCCorrelatedLCTDigi::ALCTCLCT << "]";
      errors++;
    }
  }

  // GEM-CSC stations can send out GEM-type LCTs ONLY when the ILT is turned on!
  if ((isME11_ and !runME11ILT_) or (isME21_ and !runME21ILT_)) {
    if (lct.getType() != CSCCorrelatedLCTDigi::ALCTCLCT) {
      edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid type (SIM) with GEM-CSC trigger not on: "
                                         << lct.getType() << "; allowed [" << CSCCorrelatedLCTDigi::ALCTCLCT << "]";
      errors++;
    }
  }

  // GEM-CSC types must have at least one valid GEM hit
  if ((lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCTGEM or lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCT2GEM or
       lct.getType() == CSCCorrelatedLCTDigi::ALCT2GEM or lct.getType() == CSCCorrelatedLCTDigi::CLCT2GEM) and
      !lct.getGEM1().isValid() and !lct.getGEM2().isValid()) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with valid GEM-CSC type (SIM) has no valid GEM hits: "
                                       << lct.getType();
    errors++;
  }

  // LCT type does not agree with the LCT quality when CCLUT is on
  if (runCCLUT_) {
    const bool ME11ILT(isME11_ and runME11ILT_);
    const bool ME21ILT(isME21_ and runME21ILT_);

    // GEM-CSC cases
    if (ME11ILT or ME21ILT) {
      const bool case1(lct.getType() == CSCCorrelatedLCTDigi::ALCT2GEM and
                       lct.getQuality() == static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::ALCT_2GEM));
      const bool case2(lct.getType() == CSCCorrelatedLCTDigi::CLCT2GEM and
                       lct.getQuality() == static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::CLCT_2GEM));
      const bool case3(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCTGEM and
                       lct.getQuality() ==
                           static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::ALCT_CLCT_1GEM_CSCBend));
      const bool case4(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCTGEM and
                       lct.getQuality() ==
                           static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::ALCT_CLCT_1GEM_GEMCSCBend));
      const bool case5(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCT2GEM and
                       lct.getQuality() ==
                           static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::ALCT_CLCT_2GEM_CSCBend));
      const bool case6(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCT2GEM and
                       lct.getQuality() ==
                           static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::ALCT_CLCT_2GEM_GEMCSCBend));
      const bool case7(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCT and
                       lct.getQuality() == static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::ALCT_CLCT));

      if (!(case1 or case2 or case3 or case4 or case5 or case6 or case7)) {
        edm::LogError("LCTQualityControl")
            << "CSCCorrelatedLCTDigi with valid GEM-CSC type (SIM) has no matching Run-3 quality: " << lct.getType()
            << " " << lct.getQuality();
        errors++;
      }
    }

    // regular cases
    else {
      const bool case1(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCT and
                       lct.getQuality() == static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3::LowQ));
      const bool case2(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCT and
                       lct.getQuality() == static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3::MedQ));
      const bool case3(lct.getType() == CSCCorrelatedLCTDigi::ALCTCLCT and
                       lct.getQuality() == static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3::HighQ));
      if (!(case1 or case2 or case3)) {
        edm::LogError("LCTQualityControl")
            << "CSCCorrelatedLCTDigi with invalid CSC type (SIM) has no matching Run-3 quality: " << lct.getType()
            << " " << lct.getQuality();
        errors++;
      }
    }
  }
  reportErrors(lct, errors);
}

void LCTQualityControl::checkMultiplicityBX(const std::vector<CSCALCTDigi>& collection) const {
  checkMultiplicityBX(collection, CSCConstants::MAX_ALCTS_READOUT);
}

void LCTQualityControl::checkMultiplicityBX(const std::vector<CSCCLCTDigi>& collection) const {
  checkMultiplicityBX(collection, CSCConstants::MAX_CLCTS_READOUT);
}

void LCTQualityControl::checkMultiplicityBX(const std::vector<CSCCorrelatedLCTDigi>& collection) const {
  checkMultiplicityBX(collection, CSCConstants::MAX_LCTS_PER_CSC);
}

int LCTQualityControl::getSlopePhase1(unsigned pattern) const {
  // PID 2 is actually a left-bending pattern with a negative slope
  // PID 3 is actually a right-bending pattern with a positive slope
  int slopeList[CSCConstants::NUM_CLCT_PATTERNS] = {0, 0, -8, 8, -6, 6, -4, 4, -2, 2, 0};
  return slopeList[pattern];
}

std::pair<int, int> LCTQualityControl::get_csc_clct_min_max_slope() const {
  int min_slope, max_slope;
  // Run-3 case with CCLUT
  // 5-bit number (includes the L/R bending)
  if (runCCLUT_) {
    min_slope = 0;
    max_slope = 15;
  }
  // Run-1 or Run-2 case
  // Run-3 case without CCLUT
  else {
    min_slope = -10;
    max_slope = 10;
  }

  return std::make_pair(min_slope, max_slope);
}

// Number of halfstrips and wiregroups
// +----------------------------+------------+------------+
// | Chamber type               | Num of     | Num of     |
// |                            | halfstrips | wiregroups |
// +----------------------------+------------+------------+
// | ME1/1a                     | 96         | 48         |
// | ME1/1b                     | 128        | 48         |
// | ME1/2                      | 160        | 64         |
// | ME1/3                      | 128        | 32         |
// | ME2/1                      | 160        | 112        |
// | ME3/1, ME4/1               | 160        | 96         |
// | ME2/2, ME3/2, ME4/2        | 160        | 64         |
// +----------------------------+------------+------------+

unsigned LCTQualityControl::get_csc_max_wiregroup(unsigned station, unsigned ring) const {
  unsigned max_wiregroup = 0;
  if (station == 1 && ring == 4) {  // ME1/1a
    max_wiregroup = CSCConstants::NUM_WIREGROUPS_ME11;
  } else if (station == 1 && ring == 1) {  // ME1/1b
    max_wiregroup = CSCConstants::NUM_WIREGROUPS_ME11;
  } else if (station == 1 && ring == 2) {  // ME1/2
    max_wiregroup = CSCConstants::NUM_WIREGROUPS_ME12;
  } else if (station == 1 && ring == 3) {  // ME1/3
    max_wiregroup = CSCConstants::NUM_WIREGROUPS_ME13;
  } else if (station == 2 && ring == 1) {  // ME2/1
    max_wiregroup = CSCConstants::NUM_WIREGROUPS_ME21;
  } else if (station >= 3 && ring == 1) {  // ME3/1, ME4/1
    max_wiregroup = CSCConstants::NUM_WIREGROUPS_ME31;
  } else if (station >= 2 && ring == 2) {  // ME2/2, ME3/2, ME4/2
    max_wiregroup = CSCConstants::NUM_WIREGROUPS_ME22;
  }
  return max_wiregroup;
}

unsigned LCTQualityControl::get_csc_max_halfstrip(unsigned station, unsigned ring) const {
  unsigned max_half_strip = 0;
  // ME1/1a
  if (station == 1 && ring == 4 and gangedME1a_) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME1A_GANGED;
  } else if (station == 1 && ring == 4 and !gangedME1a_) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME1A_UNGANGED;
  }
  // ME1/1b
  // In the CSC local trigger
  // ME1/a is taken together with ME1/b
  else if (station == 1 && ring == 1 and gangedME1a_) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME11_GANGED;
  } else if (station == 1 && ring == 1 and !gangedME1a_) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME11_UNGANGED;
  }
  // ME1/2
  else if (station == 1 && ring == 2) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME12;
  }
  // ME1/3
  else if (station == 1 && ring == 3) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME13;
  }
  // ME2/1
  else if (station == 2 && ring == 1) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME21;
  }
  // ME3/1, ME4/1
  else if (station >= 3 && ring == 1) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME31;
  }
  // ME2/2, ME3/2, ME4/2
  else if (station >= 2 && ring == 2) {
    max_half_strip = CSCConstants::NUM_HALF_STRIPS_ME22;
  }
  return max_half_strip;
}

unsigned LCTQualityControl::get_csc_max_quartstrip(unsigned station, unsigned ring) const {
  return get_csc_max_halfstrip(station, ring) * 2;
}

unsigned LCTQualityControl::get_csc_max_eighthstrip(unsigned station, unsigned ring) const {
  return get_csc_max_halfstrip(station, ring) * 4;
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_min_max_cfeb() const {
  // counts at 0!
  unsigned min_cfeb = 0;
  unsigned max_cfeb = 0;

  // ME1/1a
  if (theStation == 1 && theRing == 4 and gangedME1a_) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME1A_GANGED;
  } else if (theStation == 1 && theRing == 4 and !gangedME1a_) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME1A_UNGANGED;
  }
  // ME1/1b
  // In the CSC local trigger
  // ME1/a is taken together with ME1/b
  else if (theStation == 1 && theRing == 1 and gangedME1a_) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME11_GANGED;
  } else if (theStation == 1 && theRing == 1 and !gangedME1a_) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME11_UNGANGED;
  }
  // ME1/2
  else if (theStation == 1 && theRing == 2) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME12;
  }
  // ME1/3
  else if (theStation == 1 && theRing == 3) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME13;
  }
  // ME2/1
  else if (theStation == 2 && theRing == 1) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME21;
  }
  // ME3/1, ME4/1
  else if (theStation >= 3 && theRing == 1) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME31;
  }
  // ME2/2, ME3/2, ME4/2
  else if (theStation >= 2 && theRing == 2) {
    max_cfeb = CSCConstants::NUM_CFEBS_ME22;
  }
  return std::make_pair(min_cfeb, max_cfeb - 1);
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_min_max_pattern() const { return std::make_pair(2, 10); }

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_min_max_pattern_run3() const { return std::make_pair(0, 4); }

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_lct_min_max_pattern() const {
  unsigned min_pattern, max_pattern;
  // Run-1 or Run-2 case
  if (!runCCLUT_) {
    min_pattern = 2;
    max_pattern = 10;
  }
  // Run-3 case: pattern Id is interpreted as the slope!
  else {
    min_pattern = 0;
    max_pattern = 15;
  }
  return std::make_pair(min_pattern, max_pattern);
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_alct_min_max_quality() const {
  // quality is number of layers - 3
  // at least 4 layers for CSC-only trigger
  unsigned min_quality = 1;
  if (isME21_ and runME21ILT_) {
    // at least 3 layers for GEM-CSC trigger in ME2/1
    min_quality = 0;
  }
  return std::make_pair(min_quality, 3);
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_clct_min_max_quality() const {
  // quality is number of layers
  unsigned min_quality = 4;
  if ((runME11ILT_ and isME11_) or (runME21ILT_ and isME21_)) {
    // at least 3 layers for GEM-CSC trigger in ME1/1 or ME2/1
    min_quality = 3;
  }
  return std::make_pair(min_quality, 6);
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_lct_min_max_quality(unsigned station, unsigned ring) const {
  // Run-1 or Run-2
  unsigned min_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun2::HQ_PATTERN_2_3);
  unsigned max_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun2::HQ_PATTERN_10);

  const bool GEMCSC = (isME11_ and runME11ILT_) or (isME21_ and runME21ILT_);

  // Run-3
  if (run3_ and !GEMCSC) {
    min_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3::LowQ);
    max_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3::HighQ);
  }

  // Run-3 with GEM-CSC on (low-quality CLCTs are permitted, but use Run-2 data format)
  if (!runCCLUT_ and GEMCSC) {
    min_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun2::HQ_ANODE_MARGINAL_CATHODE);
    max_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun2::HQ_PATTERN_10);
  }

  // Run-3 CSC with GEM-CSC on and CCLUT on
  if (runCCLUT_ and GEMCSC) {
    min_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::CLCT_2GEM);
    max_quality = static_cast<unsigned>(LCTQualityAssignment::LCT_QualityRun3GEM::ALCT_CLCT_2GEM_GEMCSCBend);
  }
  return std::make_pair(min_quality, max_quality);
}
