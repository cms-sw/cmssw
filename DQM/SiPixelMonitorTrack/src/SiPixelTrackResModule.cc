#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResModule.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <boost/cstdint.hpp>
#include <string>
#include <stdlib.h>


// constructors
//
SiPixelTrackResModule::SiPixelTrackResModule() {
}


SiPixelTrackResModule::SiPixelTrackResModule(uint32_t id): id_(id) { 
}


// destructor
//
SiPixelTrackResModule::~SiPixelTrackResModule() { 
}


// book histograms
//
void SiPixelTrackResModule::book() {
  char hkey[80];  

  DaqMonitorBEInterface* _dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  sprintf(hkey, "hitResidual-x_module%i",id_);
  meHitResidualX_ = _dbe->book1D(hkey,"Hit Residual in X",1000,-5.,5.);

  sprintf(hkey, "hitResidual-y_module%i",id_);
  meHitResidualY_ = _dbe->book1D(hkey,"Hit Residual in Y",1000,-5.,5.);
}


// fill histograms
//
void SiPixelTrackResModule::fill(const Measurement2DVector& hitRes) {
  (meHitResidualX_)->Fill(hitRes.x()); 
  (meHitResidualY_)->Fill(hitRes.y()); 
}
