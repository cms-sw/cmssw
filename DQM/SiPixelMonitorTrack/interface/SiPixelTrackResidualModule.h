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
// $Id: SiPixelTrackResidualModule.h,v 1.1 2008/07/25 20:40:51 schuang Exp $


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

   void book(const edm::ParameterSet&, int type=0);
   void fill(const Measurement2DVector&, bool modon=true, bool ladon=true, bool layon=true, bool phion = true, bool bladeon=true, bool diskon=true, bool ringon=true);

  
  private:
    uint32_t id_; 
    bool bBookTracks;

    //MonitorElement* meNofTracks_;

    MonitorElement* meResidualX_;
    MonitorElement* meResidualY_;

    //barrel
    MonitorElement* meResidualXLad_;
    MonitorElement* meResidualYLad_;

    MonitorElement* meResidualXLay_;
    MonitorElement* meResidualYLay_;

    MonitorElement* meResidualXPhi_;
    MonitorElement* meResidualYPhi_;

    //forward
    MonitorElement* meResidualXBlade_;
    MonitorElement* meResidualYBlade_;

    MonitorElement* meResidualXDisk_;
    MonitorElement* meResidualYDisk_;

    MonitorElement* meResidualXRing_;
    MonitorElement* meResidualYRing_;


};

#endif
