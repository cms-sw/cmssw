#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

//SiStripPedestalsService
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsService.h"

#include <vector>


class SiStripZeroSuppressor {
public:
  
  SiStripZeroSuppressor(){};
  ~SiStripZeroSuppressor(){};
  
  void setSiStripPedestalsService( const SiStripPedestalsService& in ){ SiStripPedestalsService_=&in;}
  void suppress(const edm::DetSet<SiStripRawDigi>&,edm::DetSet<SiStripDigi>&){};
  void suppress(const std::vector<int16_t>&,edm::DetSet<SiStripDigi>&){};

private:

  const SiStripPedestalsService* SiStripPedestalsService_; 
};
#endif
