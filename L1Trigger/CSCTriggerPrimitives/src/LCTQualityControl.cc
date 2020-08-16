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

// Check if the ALCT is valid
void LCTQualityControl::checkValid(const CSCALCTDigi& alct, unsigned max_stubs) const {
  const unsigned max_wire = get_csc_max_wire(theStation, theRing);
  const unsigned max_quality = get_csc_alct_max_quality(theStation, theRing, runME21ILT_);

  unsigned errors = 0;

  // stub must be valid
  if (!alct.isValid()) {
    edm::LogError("LCTQualityControl") << "CSCALCTDigi with invalid bit set: " << alct.isValid();
    errors++;
  }

  // ALCT number is 1 or 2
  if (alct.getTrknmb() < 1 or alct.getTrknmb() > max_stubs) {
    edm::LogError("LCTQualityControl") << "CSCALCTDigi with invalid track number: " << alct.getTrknmb()
                                       << "; allowed [1," << max_stubs << "]";
    errors++;
  }

  // ALCT quality must be valid
  // number of layers - 3
  if (alct.getQuality() <= 0 or alct.getQuality() > max_quality) {
    edm::LogError("LCTQualityControl") << "CSCALCTDigi with invalid quality: " << alct.getQuality() << "; allowed [0,"
                                       << max_quality << "]";
    errors++;
  }

  // ALCT key wire-group must be within bounds
  if (alct.getKeyWG() > max_wire) {
    edm::LogError("LCTQualityControl") << "CSCALCTDigi with invalid wire-group: " << alct.getKeyWG() << "; allowed [0, "
                                       << max_wire << "]";
    errors++;
  }

  // ALCT with out-of-time BX
  if (alct.getBX() > CSCConstants::MAX_ALCT_TBINS - 1) {
    edm::LogError("LCTQualityControl") << "CSCALCTDigi with invalid BX: " << alct.getBX() << "; allowed [0, "
                                       << CSCConstants::MAX_LCT_TBINS - 1 << "]";
    errors++;
  }

  // ALCT is neither accelerator or collision
  if (alct.getCollisionB() > 1) {
    edm::LogError("LCTQualityControl") << "CSCALCTDigi with invalid accel/coll bit: " << alct.getCollisionB()
                                       << "; allowed [0,1]";
    errors++;
  }

  if (errors > 0) {
    edm::LogError("LCTQualityControl") << "Faulty ALCT: " << cscId_ << " " << alct << "\n errors " << errors;
  }
}

// Check if the CLCT is valid
void LCTQualityControl::checkValid(const CSCCLCTDigi& clct, unsigned max_stubs) const {
  const unsigned max_strip = get_csc_max_halfstrip(theStation, theRing);
  const auto& [min_pattern, max_pattern] = get_csc_min_max_pattern(use_run3_patterns_);
  const auto& [min_slope, max_slope] = get_csc_clct_min_max_slope(use_run3_patterns_, use_comparator_codes_);
  const auto& [min_cfeb, max_cfeb] = get_csc_min_max_cfeb(theStation, theRing);
  const unsigned max_quality = get_csc_clct_max_quality();
  unsigned errors = 0;

  // CLCT must be valid
  if (!clct.isValid()) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid bit set: " << clct.isValid();
    errors++;
  }

  // CLCT number is 1 or max
  if (clct.getTrknmb() < 1 or clct.getTrknmb() > max_stubs) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid track number: " << clct.getTrknmb()
                                       << "; allowed [1," << max_stubs << "]";
    errors++;
  }

  // CLCT quality must be valid
  // CLCTs require at least 4 layers hit
  // Run-3: ME1/1 CLCTs require only 3 layers
  // Run-4: ME2/1 CLCTs require only 3 layers
  if (clct.getQuality() < nplanes_clct_hit_pattern or clct.getQuality() > max_quality) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid quality: " << clct.getQuality() << "; allowed [0,"
                                       << max_quality << "]";
    errors++;
  }

  // CLCT half-strip must be within bounds
  if (clct.getStrip() >= CSCConstants::NUM_HALF_STRIPS_PER_CFEB) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid half-strip: " << clct.getStrip() << "; allowed [0, "
                                       << CSCConstants::NUM_HALF_STRIPS_PER_CFEB - 1 << "]";
    errors++;
  }

  // CLCT key half-strip must be within bounds
  if (clct.getKeyStrip() >= max_strip) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid key half-strip: " << clct.getKeyStrip()
                                       << "; allowed [0, " << max_strip - 1 << "]";
    errors++;
  }

  // CLCT with out-of-time BX
  if (clct.getBX() >= CSCConstants::MAX_CLCT_TBINS) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid BX: " << clct.getBX() << "; allowed [0, "
                                       << CSCConstants::MAX_CLCT_TBINS - 1 << "]";
    errors++;
  }

  // CLCT with neither left nor right bending
  if (clct.getBend() > 1) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid bending: " << clct.getBend() << "; allowed [0,1]";
    errors++;
  }

  // CLCT with an invalid pattern ID
  if (clct.getPattern() < min_pattern or clct.getPattern() > max_pattern) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid pattern ID: " << clct.getPattern() << "; allowed ["
                                       << min_pattern << ", " << max_pattern << "]";
    errors++;
  }

  if (clct.getSlope() < min_slope or clct.getSlope() > max_slope) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid slope: " << clct.getSlope() << "; allowed ["
                                       << min_slope << ", " << max_slope << "]";
    errors++;
  }

  // CLCT with an invalid CFEB ID
  if (clct.getCFEB() < min_cfeb or clct.getCFEB() > max_cfeb) {
    edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid CFEB ID: " << clct.getCFEB() << "; allowed ["
                                       << min_cfeb << ", " << max_cfeb << "]";
    errors++;
  }

  if (use_comparator_codes_) {
    // CLCT comparator code is invalid
    if (clct.getCompCode() < 0 or clct.getCompCode() >= std::pow(2, 12)) {
      edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid comparator code: " << clct.getCompCode()
                                         << "; allowed [0, " << std::pow(2, 12) - 1 << "]";
      errors++;
    }

    unsigned max_quartstrip = get_csc_max_quartstrip(theStation, theRing);
    unsigned max_eightstrip = get_csc_max_eightstrip(theStation, theRing);

    // CLCT key quart-strip must be within bounds
    if (clct.getKeyStrip(4) >= max_quartstrip) {
      edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid key quart-strip: " << clct.getKeyStrip(4)
                                         << "; allowed [0, " << max_quartstrip - 1 << "]";
      errors++;
    }

    // CLCT key eight-strip must be within bounds
    if (clct.getKeyStrip(8) >= max_eightstrip) {
      edm::LogError("LCTQualityControl") << "CSCLCTDigi with invalid key eight-strip: " << clct.getKeyStrip(8)
                                         << "; allowed [0, " << max_eightstrip - 1 << "]";
      errors++;
    }
  }

  if (errors > 0) {
    edm::LogError("LCTQualityControl") << "Faulty CLCT: " << cscId_ << " " << clct << "\n errors " << errors;
  }
}

void LCTQualityControl::checkValid(const CSCCorrelatedLCTDigi& lct) const { checkValid(lct, theStation, theRing); }

void LCTQualityControl::checkValid(const CSCCorrelatedLCTDigi& lct, const unsigned station, const unsigned ring) const {
  const unsigned max_strip = get_csc_max_halfstrip(station, ring);
  const unsigned max_quartstrip = get_csc_max_quartstrip(station, ring);
  const unsigned max_eightstrip = get_csc_max_eightstrip(station, ring);
  const unsigned max_wire = get_csc_max_wire(station, ring);
  const auto& [min_pattern, max_pattern] = get_csc_lct_min_max_pattern(use_run3_patterns_);
  const unsigned max_quality = get_csc_lct_max_quality();

  unsigned errors = 0;

  // LCT must be valid
  if (!lct.isValid()) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid bit set: " << lct.isValid();
    errors++;
  }

  // LCT number is 1 or 2
  if (lct.getTrknmb() < 1 or lct.getTrknmb() > 2) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid track number: " << lct.getTrknmb()
                                       << "; allowed [1,2]";
    errors++;
  }

  // LCT quality must be valid
  if (lct.getQuality() > max_quality) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid quality: " << lct.getQuality()
                                       << "; allowed [0,15]";
    errors++;
  }

  // LCT key half-strip must be within bounds
  if (lct.getStrip() > max_strip) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid half-strip: " << lct.getStrip()
                                       << "; allowed [0, " << max_strip << "]";
    errors++;
  }

  // LCT key quart-strip must be within bounds
  if (lct.getStrip(4) >= max_quartstrip) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid key quart-strip: " << lct.getStrip(4)
                                       << "; allowed [0, " << max_quartstrip - 1 << "]";
    errors++;
  }

  // LCT key eight-strip must be within bounds
  if (lct.getStrip(8) >= max_eightstrip) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid key eight-strip: " << lct.getStrip(8)
                                       << "; allowed [0, " << max_eightstrip - 1 << "]";
    errors++;
  }

  // LCT key wire-group must be within bounds
  if (lct.getKeyWG() > max_wire) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid wire-group: " << lct.getKeyWG()
                                       << "; allowed [0, " << max_wire << "]";
    errors++;
  }

  // LCT with out-of-time BX
  if (lct.getBX() > CSCConstants::MAX_LCT_TBINS - 1) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid BX: " << lct.getBX() << "; allowed [0, "
                                       << CSCConstants::MAX_LCT_TBINS - 1 << "]";
    errors++;
  }

  // LCT with neither left nor right bending
  if (lct.getBend() > 1) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid bending: " << lct.getBend()
                                       << "; allowed [0,1";
    errors++;
  }

  // LCT with invalid MPC link
  if (lct.getMPCLink() > CSCConstants::MAX_LCTS_PER_MPC) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid MPC link: " << lct.getMPCLink()
                                       << "; allowed [0," << CSCConstants::MAX_LCTS_PER_MPC << "]";
    errors++;
  }

  // LCT with invalid CSCID
  if (lct.getCSCID() < CSCTriggerNumbering::minTriggerCscId() or
      lct.getCSCID() > CSCTriggerNumbering::maxTriggerCscId()) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid CSCID: " << lct.getBend() << "; allowed ["
                                       << CSCTriggerNumbering::minTriggerCscId() << ", "
                                       << CSCTriggerNumbering::maxTriggerCscId() << "]";
    errors++;
  }

  // LCT with an invalid pattern ID
  if (lct.getPattern() < min_pattern or lct.getPattern() > max_pattern) {
    edm::LogError("LCTQualityControl") << "CSCCorrelatedLCTDigi with invalid pattern ID: " << lct.getPattern()
                                       << "; allowed [" << min_pattern << ", " << max_pattern << "]";
    errors++;
  }

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

  if (errors > 0) {
    edm::LogError("LCTQualityControl") << "Faulty LCT: " << cscId_ << " " << lct << "\n errors " << errors;
  }
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
  int slopeList[CSCConstants::NUM_CLCT_PATTERNS] = {0, 0, 8, -8, 6, -6, 4, -4, 2, -2, 0};
  return slopeList[pattern];
}

std::pair<int, int> LCTQualityControl::get_csc_clct_min_max_slope(bool isRun3, bool runCCLUT) const {
  int min_slope, max_slope;
  // Run-3 case with CCLUT
  if (runCCLUT) {
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

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_min_max_pattern(bool isRun3) const {
  int min_pattern, max_pattern;
  // Run-1 or Run-2 case
  if (!isRun3) {
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

std::pair<unsigned, unsigned> LCTQualityControl::get_csc_lct_min_max_pattern(bool isRun3) const {
  int min_pattern, max_pattern;
  // Run-1 or Run-2 case
  if (!isRun3) {
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
