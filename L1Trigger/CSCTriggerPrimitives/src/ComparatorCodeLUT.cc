#include "L1Trigger/CSCTriggerPrimitives/interface/ComparatorCodeLUT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ComparatorCodeLUT::ComparatorCodeLUT(const edm::ParameterSet& conf) {
  positionLUTFiles_ = conf.getParameter<std::vector<std::string>>("positionLUTFiles");
  slopeLUTFiles_ = conf.getParameter<std::vector<std::string>>("slopeLUTFiles");
  patternConversionLUTFiles_ = conf.getParameter<std::vector<std::string>>("patternConversionLUTFiles");

  for (int i = 0; i < 5; ++i) {
    lutpos_[i] = std::make_unique<CSCLUTReader>(positionLUTFiles_[i]);
    lutslope_[i] = std::make_unique<CSCLUTReader>(slopeLUTFiles_[i]);
    lutpatconv_[i] = std::make_unique<CSCLUTReader>(patternConversionLUTFiles_[i]);
  }

  clct_pattern_ = CSCPatternBank::clct_pattern_run3_;
}

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
  const unsigned positionCC(lutpos_[pattern]->lookup(comparatorCode));
  const unsigned slopeCC(lutslope_[pattern]->lookup(comparatorCode));
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

int ComparatorCodeLUT::calculateComparatorCode(const pattern& halfStripPattern) const {
  int id = 0;

  for (unsigned int column = 0; column < CSCConstants::NUM_LAYERS; column++) {
    int rowPat = 0;   //physical arrangement of the three bits
    int rowCode = 0;  //code used to identify the arrangement

    //use Firmware definition for comparator code definition
    for (int row = 2; row >= 0; row--) {
      rowPat = rowPat << 1;  //bitshift the last number to the left
      rowPat += halfStripPattern[column][row];
    }
    switch (rowPat) {
      case 0:  //000
        rowCode = 0;
        break;
      case 1:  //00X
        rowCode = 1;
        break;
      case 2:  //0X0
        rowCode = 2;
        break;
      case 4:  //00X
        rowCode = 3;
        break;
      default:
        // default return value is -1
        return -1;
    }
    //each column has two bits of information, largest layer is most significant bit
    id += (rowCode << 2 * column);
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
