// Package:    SiPixelMonitorTrack
// Class:      SiPixelResidualModule
// 
// class SiPixelResidualModule SiPixelResidualModule.h 
//       DQM/SiPixelMonitorTrack/src/SiPixelResidualModule.h
//
// Description: SiPixel hit-to-track residual data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelResidualModule.h,v 1.3 2007/05/24 06:11:46 schuang Exp $


#ifndef SiPixelMonitorTrack_SiPixelResidualModule_h
#define SiPixelMonitorTrack_SiPixelResidualModule_h


#include <boost/cstdint.hpp>

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class SiPixelResidualModule { 
  public:
    SiPixelResidualModule();
    SiPixelResidualModule(const uint32_t);
   ~SiPixelResidualModule();

    void book(const edm::ParameterSet&);
    void fill(const Measurement2DVector&);
  
  private:
    uint32_t id_; 

    MonitorElement* meResidualX_;
    MonitorElement* meResidualY_;
};

#endif
