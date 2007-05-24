#include <boost/cstdint.hpp>
#include <string>

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelResidualModule.h"


SiPixelResidualModule::SiPixelResidualModule() : id_(0) {
}


SiPixelResidualModule::SiPixelResidualModule(uint32_t id) : id_(id) { 
}


SiPixelResidualModule::~SiPixelResidualModule() { 
}


void SiPixelResidualModule::book() {
  DaqMonitorBEInterface* _dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  char hisID[80]; 

  sprintf(hisID,"residual_x_module_%i",id_);
  meResidualX_ = _dbe->book1D(hisID,"Hit-to-Track Residual in X",500,-5.,5.);

  sprintf(hisID,"residual_y_module_%i",id_);
  meResidualY_ = _dbe->book1D(hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
}


void SiPixelResidualModule::fill(const Measurement2DVector& residual) {
  (meResidualX_)->Fill(residual.x()); 
  (meResidualY_)->Fill(residual.y()); 
}
