#ifndef CalibTracker_SiPixelESProducers_SiPixelGainCalibrationService_H
#define CalibTracker_SiPixelESProducers_SiPixelGainCalibrationService_H

// ************************************************************************
// ************************************************************************
// *******     SiPixelOfflineCalibrationService                     *******
// *******     Author:   Evan Friis (evan.friis@cern.ch)            *******
// *******                                                          *******
// *******     Retrives gain calibration data from offline DB       *******
// *******     at highest (gain:pixel, pedestal:pixel) granularity  *******
// *******                                                          *******
// ************************************************************************
// ************************************************************************

// Gain Calibration base class
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"

class SiPixelGainCalibrationService : public SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibration,SiPixelGainCalibrationRcd>
{

 public:
  explicit SiPixelGainCalibrationService(const edm::ParameterSet& conf) : SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibration,SiPixelGainCalibrationRcd>(conf){};
  ~SiPixelGainCalibrationService() override{};

  // pixel granularity
  float   getPedestal  ( const uint32_t& detID,const int& col, const int& row) override;
  float   getGain      ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isDead       ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isDeadColumn ( const uint32_t& detID,const int& col, const int& row) override; //throws exception!
  bool    isNoisy       ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isNoisyColumn ( const uint32_t& detID,const int& col, const int& row) override;


};
#endif
