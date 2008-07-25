// Package:    SiPixelMonitorTrack
// Class:      SiPixelTrackResidualModule
// 
// class SiPixelTrackResidualModule SiPixelTrackResidualModule.h 
//       DQM/SiPixelMonitorTrack/src/SiPixelTrackResidualModule.h
//
// Description: SiPixel hit-to-track residual data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelTrackResidualModule.h,v 1.2 2007/06/11 18:29:02 schuang Exp $


#ifndef SiPixelMonitorTrack_SiPixelTrackResidualModule_h
#define SiPixelMonitorTrack_SiPixelTrackResidualModule_h


#include <boost/cstdint.hpp>

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class SiPixelTrackResidualModule { 
  public:
    SiPixelTrackResidualModule();
    SiPixelTrackResidualModule(const uint32_t);
   ~SiPixelTrackResidualModule();

    void book(const edm::ParameterSet&);
    void fill(const Measurement2DVector&);
  
  private:
    uint32_t id_; 

    MonitorElement* meResidualX_;
    MonitorElement* meResidualY_;
};

#endif
