#ifndef CondTools_SiPixel_SiPixelGainCalibrationService_H
#define CondTools_SiPixel_SiPixelGainCalibrationService_H

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
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationServiceBase.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"

class SiPixelGainCalibrationService : public SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibration,SiPixelGainCalibrationRcd>
{

 public:
  explicit SiPixelGainCalibrationService(const edm::ParameterSet& conf) : SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibration,SiPixelGainCalibrationRcd>(conf){};
  ~SiPixelGainCalibrationService(){};

  // pixel granularity
  float   getPedestal (const uint32_t& detID,const int& col, const int& row) {return this->getPedestalByPixel(detID, col, row);};
  float   getGain     (const uint32_t& detID,const int& col, const int& row) {return this->getGainByPixel(detID, col, row);};


};
#endif
