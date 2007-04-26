#ifndef SiPixelMonitorTrack_SiPixelTrackResModule_h
#define SiPixelMonitorTrack_SiPixelTrackResModule_h


#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <boost/cstdint.hpp>


class SiPixelTrackResModule { 
  public:
    SiPixelTrackResModule();
    SiPixelTrackResModule(uint32_t id);
   ~SiPixelTrackResModule();

    void book();
    void fill(const Measurement2DVector&);
  
  private:
    uint32_t id_; 
 
    MonitorElement* meHitResidualX_;
    MonitorElement* meHitResidualY_;
};

#endif
