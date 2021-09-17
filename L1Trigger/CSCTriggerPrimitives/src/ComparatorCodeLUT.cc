#include "L1Trigger/CSCTriggerPrimitives/interface/ComparatorCodeLUT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ComparatorCodeLUT::ComparatorCodeLUT(const edm::ParameterSet& conf) {
  auto commonParams = conf.getParameter<edm::ParameterSet>("commonParam");
  use11BitPatterns_ = commonParams.getParameter<bool>("use11BitPatterns");

  clct_pattern_ = CSCPatternBank::clct_pattern_run3_;
  if (use11BitPatterns_) {
    clct_pattern_ = CSCPatternBank::clct_pattern_run3_11bit_;
  }
}

void ComparatorCodeLUT::setESLookupTables(const CSCL1TPLookupTableCCLUT* conf) { lookupTableCCLUT_ = conf; }

void ComparatorCodeLUT::run(CSCCLCTDigi& digi, unsigned numCFEBs) const {
  // print out the old CLCT for debugging
  if (infoV_ > 2) {
    std::ostringstream strm;
    strm << "\n";
    strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    strm << "+                  Before CCCLUT algorithm:                       +\n";
    strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    strm << " Old CLCT digi " << digi << "\n";
    strm << " 1/4 strip bit " << digi.getQuartStripBit() << " 1/8 strip bit " << digi.getEighthStripBit() << "\n";
    strm << " 1/4 strip number " << digi.getKeyStrip(4) << " 1/8 strip number " << digi.getKeyStrip(8) << "\n";
    strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    LogDebug("ComparatorCodeLUT") << strm.str();
  }

  // set Run-3 flag
  digi.setRun3(true);

  // Get the comparator hits
  auto compHits = digi.getHits();

  // Wrap the comparator code in a format for calculation
  pattern compHitsCC;

  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
    int iCC = 0;
    for (int j = 0; j < CSCConstants::CLCT_PATTERN_WIDTH; j++) {
      // only fill when the pattern is active
      if (clct_pattern_[digi.getPattern()][i][j]) {
        if (compHits[i][j] != CSCConstants::INVALID_HALF_STRIP) {
          compHitsCC[i][iCC] = 1;
        } else {
          compHitsCC[i][iCC] = 0;
        }
        iCC++;
      }
    }
  }

  // calculate the comparator code
  const int comparatorCode(calculateComparatorCode(compHitsCC));

  // store the comparator code
  digi.setCompCode(comparatorCode);

  // calculate the slope and position offset
  const int pattern(digi.getPattern());

  // set the Run-3 pattern
  digi.setRun3Pattern(pattern);

  // look-up the unsigned values
  const unsigned positionCC(lookupTableCCLUT_->cclutPosition(pattern, comparatorCode));
  const unsigned slopeCC(lookupTableCCLUT_->cclutSlope(pattern, comparatorCode));
  const unsigned run2PatternCC(convertSlopeToRun2Pattern(slopeCC));

  // if the slope is negative, set bending to 0
  const bool slopeCCSign((slopeCC >> 4) & 0x1);
  const unsigned slopeCCValue(slopeCC & 0xf);
  digi.setBend(slopeCCSign);

  // calculate the new position
  uint16_t halfstrip = digi.getKeyStrip();
  std::tuple<int16_t, bool, bool> stripoffset;
  assignPositionCC(positionCC, stripoffset);
  const int halfstripoffset = std::get<0>(stripoffset);
  halfstrip += halfstripoffset;

  // store the new CFEB, 1/2, 1/4 and 1/8 strip positions
  digi.setCFEB(halfstrip / CSCConstants::NUM_HALF_STRIPS_PER_CFEB);
  digi.setStrip(halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB);
  digi.setQuartStripBit(std::get<1>(stripoffset));
  digi.setEighthStripBit(std::get<2>(stripoffset));

  // store the bending angle value in the pattern data member
  digi.setSlope(slopeCCValue);

  // set the quasi Run-2 pattern - to accommodate integration with EMTF/OMTF
  digi.setPattern(run2PatternCC);

  // now print out the new CLCT for debugging
  if (infoV_ > 2) {
    std::ostringstream strm;
    strm << "\n";
    strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    strm << "+                  CCCLUT algorithm results:                       +\n";
    strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    strm << " New CLCT digi " << digi << "\n";
    strm << " 1/4 strip bit " << digi.getQuartStripBit() << " 1/8 strip bit " << digi.getEighthStripBit() << "\n";
    strm << " 1/4 strip number " << digi.getKeyStrip(4) << " 1/8 strip number " << digi.getKeyStrip(8) << "\n";
    strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    LogDebug("ComparatorCodeLUT") << strm.str();
  }
}

// general case
int ComparatorCodeLUT::calculateComparatorCodeLayer(int layer, const std::array<int, 3>& halfStripPattern) const {
  // case for key-layer when pattern is 11-bit
  // this is either 0 or 1
  if (use11BitPatterns_ and layer == CSCConstants::KEY_CLCT_LAYER - 1) {
    return halfStripPattern[1];
  }

  // physical arrangement of the three bits
  int hitPattern = 0;
  for (int ihit = 2; ihit >= 0; ihit--) {
    hitPattern = hitPattern << 1;  //bitshift the last number to the left
    hitPattern += halfStripPattern[ihit];
  }

  // code used to identify the arrangement
  int layerCode = 0;
  switch (hitPattern) {
    case 0:  //000
      layerCode = 0;
      break;
    case 1:  //00X
      layerCode = 1;
      break;
    case 2:  //0X0
      layerCode = 2;
      break;
    case 4:  //X00
      layerCode = 3;
      break;
    default:
      // default return value is 0
      return 0;
  }

  return layerCode;
}

int ComparatorCodeLUT::calculateComparatorCode(const pattern& halfStripPattern) const {
  int id = 0;
  int shift = 0;

  for (unsigned int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
    id += (calculateComparatorCodeLayer(layer, halfStripPattern[layer]) << shift);
    // note: the key layer is the 3rd layer, but has number "2" when starting from "0"
    // this layer has 1 bit of information when use11BitPatterns_ is on
    if (use11BitPatterns_ and layer == CSCConstants::KEY_CLCT_LAYER - 1) {
      shift += 1;
    }
    // each layer has two bits of information, largest layer is most significant bit2
    else {
      shift += 2;
    }
  }
  return id;
}

unsigned ComparatorCodeLUT::convertSlopeToRun2Pattern(const unsigned slope) const {
  const unsigned slopeList[32] = {10, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 2,
                                  10, 10, 10, 9, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3, 3, 3};
  return slopeList[slope];
}

void ComparatorCodeLUT::assignPositionCC(const unsigned offset, std::tuple<int16_t, bool, bool>& returnValue) const {
  /*
    | Value | Half-Strip Offset  | Delta Half-Strip  | Quarter-Strip Bit  | Eighth-Strip Bit |
    |-------|--------------------|-------------------|--------------------|------------------|
    |   0   |   -7/4             |   -2              |   0                |   1              |
    |   1   |   -3/2             |   -2              |   1                |   0              |
    |   2   |   -5/4             |   -2              |   1                |   1              |
    |   3   |   -1               |   -1              |   0                |   0              |
    |   4   |   -3/4             |   -1              |   0                |   1              |
    |   5   |   -1/2             |   -1              |   1                |   0              |
    |   6   |   -1/4             |   -1              |   1                |   1              |
    |   7   |   0                |   0               |   0                |   0              |
    |   8   |   1/4              |   0               |   0                |   1              |
    |   9   |   1/2              |   0               |   1                |   0              |
    |   10  |   3/4              |   0               |   1                |   1              |
    |   11  |   1                |   1               |   0                |   0              |
    |   12  |   5/4              |   1               |   0                |   1              |
    |   13  |   3/2              |   1               |   1                |   0              |
    |   14  |   7/4              |   1               |   1                |   1              |
    |   15  |   2                |   2               |   0                |   0              |
  */
  std::vector<std::tuple<int16_t, bool, bool>> my_tuple = {
      {-2, false, true},
      {-2, true, false},
      {-2, true, true},
      {-1, false, false},
      {-1, false, true},
      {-1, true, false},
      {-1, true, true},
      {0, false, false},
      {0, false, true},
      {0, true, false},
      {0, true, true},
      {1, false, false},
      {1, false, true},
      {1, true, false},
      {1, true, true},
      {2, false, false},
  };
  returnValue = my_tuple[offset];
}
