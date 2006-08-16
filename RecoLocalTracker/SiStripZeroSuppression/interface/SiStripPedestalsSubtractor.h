#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSUBTRACTOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSUBTRACTOR_H

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

//SiStripPedestalsService
#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsService.h"

#include <vector>


class SiStripPedestalsSubtractor {
public:
  
  SiStripPedestalsSubtractor(){};
  ~SiStripPedestalsSubtractor(){};

  void setSiStripPedestalsService( SiStripPedestalsService* in ){SiStripPedestalsService_=in;} 
  void subtract(const edm::DetSet<SiStripRawDigi>&, std::vector<int16_t>&);

private:

  SiStripPedestalsService* SiStripPedestalsService_; 
};
#endif
