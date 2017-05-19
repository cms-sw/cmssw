#ifndef RecoCTPPS_CTPPSPixelLocal_CTPPSPixelGainCalibrationDBService_h
#define RecoCTPPS_CTPPSPixelLocal_CTPPSPixelGainCalibrationDBService_h
// -*- C++ -*-
//
// Package:     RecoCTPPS/CTPPSPixelLocal
// Class  :     CTPPSPixelGainCalibrationDBService
// 
/**\class CTPPSPixelGainCalibrationDBService CTPPSPixelGainCalibrationDBService.h "RecoCTPPS/CTPPSPixelLocal/interface/CTPPSPixelGainCalibrationDBService.h"

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
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
class CTPPSPixelGainCalibrationDBService
{

   public:
      CTPPSPixelGainCalibrationDBService();
      virtual ~CTPPSPixelGainCalibrationDBService();
      virtual void getDB(const edm::Event& e, const edm::EventSetup& c);
      const CTPPSPixelGainCalibrations* pPixelGainCalibrations;
      const CTPPSPixelGainCalibrations* getCalibs() const {return pPixelGainCalibrations;}
   private:
      CTPPSPixelGainCalibrationDBService(const CTPPSPixelGainCalibrationDBService&); // stop default

      const CTPPSPixelGainCalibrationDBService& operator=(const CTPPSPixelGainCalibrationDBService&); // stop default

};

#endif
