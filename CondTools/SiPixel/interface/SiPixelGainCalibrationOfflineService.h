#ifndef CondTools_SiPixel_SiPixelGainCalibrationOfflineService_H
#define CondTools_SiPixel_SiPixelGainCalibrationOfflineService_H

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
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationServiceBase.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h" 
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineRcd.h"

class SiPixelGainCalibrationOfflineService : public SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationOffline,SiPixelGainCalibrationOfflineRcd>
{

 public:
  explicit SiPixelGainCalibrationOfflineService(const edm::ParameterSet& conf) : SiPixelGainCalibrationServicePayloadGetter<SiPixelGainCalibrationOffline,SiPixelGainCalibrationOfflineRcd>(conf){};
  ~SiPixelGainCalibrationOfflineService(){};

  // pixel granularity
  float   getPedestal (const uint32_t& detID,const int& col, const int& row) {return this->getPedestalByPixel(detID, col, row);};
  float   getGain     (const uint32_t& detID,const int& col, const int& row = 0) {return this->getGainByColumn(detID, col);};


};
#endif
