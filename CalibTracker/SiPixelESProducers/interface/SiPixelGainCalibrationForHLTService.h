#ifndef CalibTracker_SiPixelESProducers_SiPixelGainCalibrationForHLTService_H
#define CalibTracker_SiPixelESProducers_SiPixelGainCalibrationForHLTService_H

// ************************************************************************
// ************************************************************************
// *******     SiPixelOfflineCalibrationForHLTService               *******
// *******     Author:   Evan Friis (evan.friis@cern.ch)            *******
// *******                                                          *******
// *******     Retrives gain calibration data from offline DB       *******
// *******     at lowest  (gain:column,pedestal:column) granularity *******
// *******                                                          *******
// ************************************************************************
// ************************************************************************
//
// Gain Calibration base class
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"

class SiPixelGainCalibrationForHLTService final : public SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationForHLT,SiPixelGainCalibrationForHLTRcd>
{

 public:
  explicit SiPixelGainCalibrationForHLTService(const edm::ParameterSet& conf) : SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationForHLT,SiPixelGainCalibrationForHLTRcd>(conf){};
  ~SiPixelGainCalibrationForHLTService() override{};

  void calibrate(uint32_t detID, DigiIterator b, DigiIterator e, float conversionFactor, float offset, int * electron) override;


  // column granularity
  float   getPedestal  ( const uint32_t& detID,const int& col, const int& row) override;
  float   getGain      ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isDead       ( const uint32_t& detID,const int& col, const int& row) override; //also return dead by column.
  bool    isDeadColumn ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isNoisy       ( const uint32_t& detID,const int& col, const int& row) override;
  bool    isNoisyColumn ( const uint32_t& detID,const int& col, const int& row) override;

};
#endif
