#ifndef CondTools_SiPixel_SiPixelGainCalibrationForHLTService_H
#define CondTools_SiPixel_SiPixelGainCalibrationForHLTService_H

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
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationServiceBase.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"

class SiPixelGainCalibrationForHLTService : public SiPixelGainCalibrationServiceBase<SiPixelGainCalibrationForHLT,SiPixelGainCalibrationForHLTRcd>
{

 public:
  explicit SiPixelGainCalibrationForHLTService(const edm::ParameterSet& conf) : SiPixelGainCalibrationServiceBase<SiPixelGainCalibrationForHLT,SiPixelGainCalibrationForHLTRcd>(conf){};
  ~SiPixelGainCalibrationForHLTService(){};

  // column granularity
  float   getPedestal (const uint32_t& detID,const int& col) {return this->getPedestalByColumn(detID, col);};
  float   getGain     (const uint32_t& detID,const int& col) {return this->getGainByColumn(detID, col);};


};
#endif
