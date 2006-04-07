#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPCOMMONMODENOISESUBTRACTOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPCOMMONMODENOISESUBTRACTOR_H

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>


class SiStripCommonModeNoiseSubtractor {
public:
  
  SiStripCommonModeNoiseSubtractor(std::string mode):CMNSubMode(mode){};
  ~SiStripCommonModeNoiseSubtractor(){};
  
  void subtract(std::vector<int16_t>&);

private:

  std::string CMNSubMode;
 
};
#endif
