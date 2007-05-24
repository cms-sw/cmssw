#ifndef SiPixelMonitorTrack_SiPixelResidualModule_h
#define SiPixelMonitorTrack_SiPixelResidualModule_h


#include <boost/cstdint.hpp>

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class SiPixelResidualModule { 
  public:
    SiPixelResidualModule();
    SiPixelResidualModule(const uint32_t id);
   ~SiPixelResidualModule();

    void book();
    void fill(const Measurement2DVector&);
  
  private:
    uint32_t id_; 

    MonitorElement* meResidualX_;
    MonitorElement* meResidualY_;
};

#endif
