#ifndef CondTools_SiPixel_SiPixelGainCalibrationService_H
#define CondTools_SiPixel_SiPixelGainCalibrationService_H
// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"

class SiPixelGainCalibrationService {

 public:
  SiPixelGainCalibrationService(const edm::ParameterSet& conf);
  ~SiPixelGainCalibrationService(){};

  void    setESObjects(const edm::EventSetup& es );
  float   getPedestal (const uint32_t& detID,const int& col, const int& row) ;
  float   getGain     (const uint32_t& detID,const int& col, const int& row) ;

  std::vector<uint32_t> getDetIds();

 private:

  edm::ParameterSet conf_;
  edm::ESHandle<SiPixelGainCalibration> ped;
  bool ESetupInit_;

  uint32_t old_detID;
  int      old_cols;
  SiPixelGainCalibration::Range old_range;
};
#endif
