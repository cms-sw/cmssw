#pragma once
#include<cstdint>
#include<tuple>


// copy of SiPixelGainCalibrationForHLT
class SiPixelGainForHLTonGPU {

 public:

  struct DecodingStructure{  
    unsigned int gain :8;
    unsigned int ped  :8;
  };
  
  using Range = std::pair<uint32_t,uint32_t>;
 

  constexpr
  std::pair<float,float> getPedAndGain(uint32_t moduleInd, int col, int row, bool& isDeadColumn, bool& isNoisyColumn ) const {
    auto range = rangeAndCols[moduleInd].first;
    auto nCols = rangeAndCols[moduleInd].second;
    // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
    unsigned int lengthOfColumnData  = (range.second-range.first)/nCols;
    unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block 
    unsigned int numberOfDataBlocksToSkip = row / numberOfRowsAveragedOver_;

    auto first = v_pedestals+range.first;
    const DecodingStructure & s = (const DecodingStructure & ) *(first + col*lengthOfColumnData + lengthOfAveragedDataInEachColumn*numberOfDataBlocksToSkip);

    isDeadColumn = (s.ped & 0xFF) == deadFlag_;
    isNoisyColumn = (s.ped & 0xFF) == noisyFlag_;

    return std::make_pair(decodePed(s.ped & 0xFF),decodeGain(s.gain & 0xFF));
  }



  constexpr float decodeGain(unsigned int gain) const {return gain*gainPrecision + minGain_;}
  constexpr float decodePed (unsigned int ped) const { return ped*pedPrecision + minPed_;}

  unsigned char * v_pedestals;
  std::pair<Range, int> rangeAndCols[2000];

  float  minPed_, maxPed_, minGain_, maxGain_;

  float pedPrecision, gainPrecision;
  unsigned int numberOfRowsAveragedOver_; // this is 80!!!!
  unsigned int nBinsToUseForEncoding_;
  unsigned int deadFlag_;
  unsigned int noisyFlag_;
};
