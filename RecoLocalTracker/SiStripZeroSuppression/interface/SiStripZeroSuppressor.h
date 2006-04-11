#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//SiStripPedestalsService
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsService.h"

#include <vector>

class SiStripZeroSuppressor {
public:
  
  SiStripZeroSuppressor(uint16_t fedalgo):theFEDalgorithm(fedalgo){};
  ~SiStripZeroSuppressor(){};
  
  void setSiStripPedestalsService( SiStripPedestalsService* in ){ SiStripPedestalsService_=in;}
  void suppress(const edm::DetSet<SiStripRawDigi>&,edm::DetSet<SiStripDigi>&);
  void suppress(const std::vector<int16_t>&,edm::DetSet<SiStripDigi>&);

  bool IsAValidDigi();

private:
  uint16_t theFEDalgorithm;
  const SiStripPedestalsService* SiStripPedestalsService_; 

  int16_t adc;
  int16_t adcPrev;
  int16_t adcNext;
  int16_t adcMaxNeigh;
  int16_t adcPrev2;
  int16_t adcNext2;
  float theFEDlowThresh;
  float theFEDhighThresh;

  float thePrevFEDlowThresh;
  float thePrevFEDhighThresh;
  float theNextFEDlowThresh;
  float theNextFEDhighThresh;

  float theNeighFEDlowThresh;
  float theNeighFEDhighThresh;

  float thePrev2FEDlowThresh;
  float theNext2FEDlowThresh;

};
#endif
