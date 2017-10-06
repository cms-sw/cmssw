#ifndef CalibTracker_SiPixelESProducers_SiPixelGainCalibrationOfflineSimService_H
#define CalibTracker_SiPixelESProducers_SiPixelGainCalibrationOfflineSimService_H

// ************************************************************************
// ************************************************************************
// *******     SiPixelOfflineCalibrationOfflineSimService           *******
// *******     Author:   Freya Blekman                              *******
// *******     based on code by:   Evan Friis (evan.friis@cern.ch)  *******
// *******                                                          *******
// *******     Retrives gain calibration data from offline DB       *******
// *******     at medium  (gain:column,pedestal:pixel) granularity  *******
// *******                                                          *******
// ************************************************************************
// ************************************************************************

// Gain CalibrationOffline base class
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineSimRcd.h"

class SiPixelGainCalibrationOfflineSimService : public SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationOffline,SiPixelGainCalibrationOfflineSimRcd>
{

 public:
  explicit SiPixelGainCalibrationOfflineSimService(const edm::ParameterSet& conf) : SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationOffline,SiPixelGainCalibrationOfflineSimRcd>(conf){};
  ~SiPixelGainCalibrationOfflineSimService() override{};

  // pixel granularity
  float   getPedestal  ( const uint32_t& detID,const int& col, const int& row) override;
  float   getGain      ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isDead       ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isDeadColumn ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isNoisy       ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isNoisyColumn ( const uint32_t& detID,const int& col, const int& row) override;
};
#endif
