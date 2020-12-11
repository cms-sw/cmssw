#include "L1Trigger/CSCTriggerPrimitives/interface/LCTQualityControl.h"
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
  const unsigned max_wire = get_csc_max_wire(theStation, theRing);
  const unsigned max_quality = get_csc_alct_max_quality(theStation, theRing, runME21ILT_);

  unsigned errors = 0;

  // stub must be valid
  checkRange(alct.isValid(), 1, 1, "CSCALCTDigi with invalid bit set: ", errors);

  // ALCT number is 1 or 2
  checkRange(alct.getTrknmb(), 1, max_stubs, "CSCALCTDigi with invalid track number: ", errors);

  // ALCT quality must be valid
  // number of layers - 3
  checkRange(alct.getQuality(), 1, max_quality, "CSCALCTDigi with invalid quality: ", errors);

  // ALCT key wire-group must be within bounds
  checkRange(alct.getKeyWG(), 0, max_wire - 1, "CSCALCTDigi with invalid wire-group: ", errors);

  // ALCT with out-of-time BX
  checkRange(alct.getBX(), 0, CSCConstants::MAX_ALCT_TBINS - 1, "CSCALCTDigi with invalid BX: ", errors);

  // ALCT is neither accelerator or collision
  checkRange(alct.getCollisionB(), 0, 1, "CSCALCTDigi with invalid accel/coll biit: ", errors);

  reportErrors(alct, errors);
}

// Check if the CLCT is valid
void LCTQualityControl::checkValid(const CSCCLCTDigi& clct, unsigned max_stubs) const {
  const unsigned max_strip = get_csc_max_halfstrip(theStation, theRing);
  const auto& [min_pattern_run2, max_pattern_run2] = get_csc_min_max_pattern(false);
  const auto& [min_pattern_run3, max_pattern_run3] = get_csc_min_max_pattern(true);
  const auto& [min_slope, max_slope] = get_csc_clct_min_max_slope();
  const auto& [min_cfeb, max_cfeb] = get_csc_min_max_cfeb(theStation, theRing);
  const unsigned max_quality = get_csc_clct_max_quality();
  unsigned errors = 0;

  // CLCT must be valid
  checkRange(clct.isValid(), 1, 1, "CSCCLCTDigi with invalid bit set: ", errors);

  // CLCT number is 1 or max
  checkRange(clct.getTrknmb(), 1, max_stubs, "CSCCLCTDigi with invalid track number: ", errors);

  // CLCT quality must be valid
  // CLCTs require at least 4 layers hit
  // Run-3: ME1/1 CLCTs require only 3 layers
  // Run-4: ME2/1 CLCTs require only 3 layers
  checkRange(clct.getQuality(), nplanes_clct_hit_pattern, max_quality, "CSCCLCTDigi with invalid quality: ", errors);

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
    const unsigned max_eightstrip = get_csc_max_eightstrip(theStation, theRing);

    // CLCT key quart-strip must be within bounds
    checkRange(clct.getKeyStrip(4), 0, max_quartstrip - 1, "CSCCLCTDigi with invalid key quart-strip: ", errors);

    // CLCT key eight-strip must be within bounds
    checkRange(clct.getKeyStrip(8), 0, max_eightstrip - 1, "CSCCLCTDigi with invalid key quart-strip: ", errors);
  }

  reportErrors(clct, errors);
}

void LCTQualityControl::checkValid(const CSCCorrelatedLCTDigi& lct) const { checkValid(lct, theStation, theRing); }

void LCTQualityControl::checkValid(const CSCCorrelatedLCTDigi& lct, const unsigned station, const unsigned ring) const {
  const unsigned max_strip = get_csc_max_halfstrip(station, ring);
  const unsigned max_quartstrip = get_csc_max_quartstrip(station, ring);
  const unsigned max_eightstrip = get_csc_max_eightstrip(station, ring);
  const unsigned max_wire = get_csc_max_wire(station, ring);
  const auto& [min_pattern, max_pattern] = get_csc_lct_min_max_pattern();
  const unsigned max_quality = get_csc_lct_max_quality();

  unsigned errors = 0;

  // LCT must be valid
  checkRange(lct.isValid(), 1, 1, "CSCCorrelatedLCTDigi with invalid bit set: ", errors);

  // LCT number is 1 or 2
  checkRange(lct.getTrknmb(), 1, 2, "CSCCorrelatedLCTDigi with invalid track number: ", errors);

  // LCT quality must be valid
  checkRange(lct.getQuality(), 0, max_quality, "CSCCorrelatedLCTDigi with invalid quality: ", errors);

  // LCT key half-strip must be within bounds
  checkRange(lct.getStrip(), 0, max_strip - 1, "CSCCorrelatedLCTDigi with invalid key half-strip: ", errors);

  // LCT key quart-strip must be within bounds
  checkRange(lct.getStrip(4), 0, max_quartstrip - 1, "CSCCorrelatedLCTDigi with invalid key quart-strip: ", errors);

  // LCT key eight-strip must be within bounds
  checkRange(lct.getStrip(8), 0, max_eightstrip - 1, "CSCCorrelatedLCTDigi with invalid key eight-strip: ", errors);

  // LCT key wire-group must be within bounds
  checkRange(lct.getKeyWG(), 0, max_wire - 1, "CSCCorrelatedLCTDigi with invalid wire-group: ", errors);

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
  checkRange(lct.getPattern(), min_pattern, max_pattern, "CSCCorrelatedLCTDigi with invalid pattern ID: ", errors);

  // simulated LCT type must be valid
  if (lct.getType() == CSCCorrelatedLCTDigi::CLCTALCT or lct.getType() == CSCCorrelatedLCTDigi::CLCTONLY or
      lct.getType() == CSCCorrelatedLCTDigi::ALCTONLY) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid type (SIM): " << lct.getType()
                                       << "; allowed [" << CSCCorrelatedLCTDigi::ALCTCLCT << ", "
                                       << CSCCorrelatedLCTDigi::CLCT2GEM << "]";
    errors++;
  }

  // non-GEM-CSC stations ALWAYS send out ALCTCLCT type LCTs
  if (!(ring == 1 and (station == 1 or station == 2))) {
    if (lct.getType() != CSCCorrelatedLCTDigi::ALCTCLCT) {
      edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid type (SIM) in this station: "
                                         << lct.getType() << "; allowed [" << CSCCorrelatedLCTDigi::ALCTCLCT << "]";
      errors++;
    }
  }

  // GEM-CSC stations can send out GEM-type LCTs ONLY when the ILT is turned on!
  if (ring == 1 and lct.getType() != CSCCorrelatedLCTDigi::ALCTCLCT) {
    if ((station == 1 and !runME11ILT_) or (station == 2 and !runME21ILT_)) {
      edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid type (SIM) with GEM-CSC trigger not on: "
                                         << lct.getType() << "; allowed [" << CSCCorrelatedLCTDigi::ALCTCLCT << "]";
      errors++;
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

int LCTQualityControl::getSlopePhase1(int pattern) const {
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
    max_slope = 31;
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

unsigned LCTQualityControl::get_csc_max_wire(int station, int ring) const {
  unsigned max_wire = 0;            // wiregroup
  if (station == 1 && ring == 4) {  // ME1/1a
    max_wire = 48;
  } else if (station == 1 && ring == 1) {  // ME1/1b
    max_wire = 48;
  } else if (station == 1 && ring == 2) {  // ME1/2
    max_wire = 64;
  } else if (station == 1 && ring == 3) {  // ME1/3
    max_wire = 32;
  } else if (station == 2 && ring == 1) {  // ME2/1
    max_wire = 112;
  } else if (station >= 3 && ring == 1) {  // ME3/1, ME4/1
    max_wire = 96;
  } else if (station >= 2 && ring == 2) {  // ME2/2, ME3/2, ME4/2
    max_wire = 64;
  }
  return max_wire;
}

unsigned LCTQualityControl::get_csc_max_halfstrip(int station, int ring) const {
  int max_strip = 0;                // halfstrip
  if (station == 1 && ring == 4) {  // ME1/1a
    max_strip = 96;
  } else if (station == 1 && ring == 1) {  // ME1/1b
    // In the CSC local trigger
    // ME1/a is taken together with ME1/b
    max_strip = 128 + 96;
  } else if (station == 1 && ring == 2) {  // ME1/2
    max_strip = 160;
  } else if (station == 1 && ring == 3) {  // ME1/3
    max_strip = 128;
  } else if (station == 2 && ring == 1) {  // ME2/1
    max_strip = 160;
  } else if (station >= 3 && ring == 1) {  // ME3/1, ME4/1
    max_strip = 160;
  } else if (station >= 2 && ring == 2) {  // ME2/2, ME3/2, ME4/2
    max_strip = 160;
  }
  return max_strip;
}

unsigned LCTQualityControl::get_csc_max_quartstrip(int station, int ring) const {
  return get_csc_max_halfstrip(station, ring) * 2;
}

unsigned LCTQualityControl::get_csc_max_eightstrip(int station, int ring) const {
  return get_csc_max_halfstrip(station, ring) * 4;
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_min_max_cfeb(int station, int ring) const {
  // 5 CFEBs [0,4] for non-ME1/1 chambers
  int min_cfeb = 0;
  int max_cfeb = CSCConstants::MAX_CFEBS - 1;  // 4
  // 7 CFEBs [0,6] for ME1/1 chambers
  if (station == 1 and ring == 1) {
    max_cfeb = 6;
  }
  return std::make_pair(min_cfeb, max_cfeb);
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_min_max_pattern(bool runCCLUT) const {
  int min_pattern, max_pattern;
  // Run-1 or Run-2 case
  if (!runCCLUT) {
    min_pattern = 2;
    max_pattern = 10;
  }
  // Run-3 case
  else {
    min_pattern = 0;
    max_pattern = 4;
  }
  return std::make_pair(min_pattern, max_pattern);
}

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_lct_min_max_pattern() const {
  int min_pattern, max_pattern;
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

unsigned LCTQualityControl::get_csc_alct_max_quality(int station, int ring, bool runGEMCSC) const {
  int max_quality = 3;
  // GE2/1-ME2/1 ALCTs are allowed 3-layer ALCTs
  if (runGEMCSC and station == 2 and ring == 1) {
    max_quality = 4;
  }
  return max_quality;
}

unsigned LCTQualityControl::get_csc_clct_max_quality() const {
  int max_quality = 6;
  return max_quality;
}

unsigned LCTQualityControl::get_csc_lct_max_quality() const {
  int max_quality = 15;
  return max_quality;
}
