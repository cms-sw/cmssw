#ifndef RecoPPS_Local_CTPPSPixelGainCalibrationDBService_h
#define RecoPPS_Local_CTPPSPixelGainCalibrationDBService_h
// -*- C++ -*-
//
// Package:     RecoPPS/Local
// Class  :     CTPPSPixelGainCalibrationDBService
//
/**\class CTPPSPixelGainCalibrationDBService CTPPSPixelGainCalibrationDBService.h "RecoPPS/Local/interface/CTPPSPixelGainCalibrationDBService.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Helio Nogima
//         Created:  Thu, 23 Feb 2017 17:37:25 GMT
//
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelGainCalibrations.h"
class CTPPSPixelGainCalibrationDBService {
public:
  CTPPSPixelGainCalibrationDBService();
  virtual ~CTPPSPixelGainCalibrationDBService();
  virtual void getDB(const edm::Event& e, const edm::EventSetup& c);
  const CTPPSPixelGainCalibrations* getCalibs() const { return pPixelGainCalibrations; }

private:
  CTPPSPixelGainCalibrationDBService(const CTPPSPixelGainCalibrationDBService&) = delete;
  const CTPPSPixelGainCalibrations* pPixelGainCalibrations;
  const CTPPSPixelGainCalibrationDBService& operator=(const CTPPSPixelGainCalibrationDBService&) = delete;
};

#endif
