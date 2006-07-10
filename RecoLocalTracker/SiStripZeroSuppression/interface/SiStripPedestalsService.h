#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSERVICE_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSERVICE_H

//edm
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//ES Data
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h" 
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

class SiStripPedestalsService {

 public:
  SiStripPedestalsService(const edm::ParameterSet& conf);
  ~SiStripPedestalsService(){};
  
  //  void configure( const edm::EventSetup& es );
  void setESObjects( const edm::EventSetup& es );
  int16_t getPedestal(const uint32_t& detID,const uint16_t& strip) ;
  float   getLowTh   (const uint32_t& detID,const uint16_t& strip) ;
  float   getHighTh  (const uint32_t& detID,const uint16_t& strip) ;

 private:
  edm::ParameterSet conf_;
  edm::ESHandle<SiStripPedestals> ped;
  bool UseCalibDataFromDB_;

  double ElectronsPerADC_;
  double ENC_;
  double BadStripProbability_;
  int    PedestalValue_;
  float  LTh_;
  float  HTh_;

  SiStripPedestals::Range old_range;
  uint32_t old_detID;
  float    old_noise;

};

#endif
