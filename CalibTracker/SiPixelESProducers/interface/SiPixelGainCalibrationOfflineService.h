#ifndef CalibTracker_SiPixelESProducers_SiPixelGainCalibrationOfflineService_H
#define CalibTracker_SiPixelESProducers_SiPixelGainCalibrationOfflineService_H

// ************************************************************************
// ************************************************************************
// *******     SiPixelOfflineCalibrationOfflineService              *******
// *******     Author:   Evan Friis (evan.friis@cern.ch)            *******
// *******                                                          *******
// *******     Retrives gain calibration data from offline DB       *******
// *******     at medium  (gain:column,pedestal:pixel) granularity  *******
// *******                                                          *******
// ************************************************************************
// ************************************************************************

// Gain CalibrationOffline base class
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineRcd.h"

class SiPixelGainCalibrationOfflineService : public SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationOffline,SiPixelGainCalibrationOfflineRcd>
{

 public:
  explicit SiPixelGainCalibrationOfflineService(const edm::ParameterSet& conf) : SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationOffline,SiPixelGainCalibrationOfflineRcd>(conf){};
  ~SiPixelGainCalibrationOfflineService() override{};

  // pixel granularity
  float   getPedestal  ( const uint32_t& detID,const int& col, const int& row) override;
  float   getGain      ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isDead       ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isDeadColumn ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isNoisy       ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isNoisyColumn ( const uint32_t& detID,const int& col, const int& row) override;
};
#endif
