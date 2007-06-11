// Package:    SiPixelMonitorTrack
// Class:      SiPixelResidualModule
// 
// class SiPixelResidualModule SiPixelResidualModule.cc 
//       DQM/SiPixelMonitorTrack/src/SiPixelResidualModule.cc
//
// Description: SiPixel hit-to-track residual data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelResidualModule.cc,v 1.3 2007/05/24 06:11:46 schuang Exp $


#include <boost/cstdint.hpp>
#include <string>
#include <iostream>

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelResidualModule.h"


using namespace std; 


SiPixelResidualModule::SiPixelResidualModule() : id_(0) {
}


SiPixelResidualModule::SiPixelResidualModule(uint32_t id) : id_(id) { 
}


SiPixelResidualModule::~SiPixelResidualModule() { 
}


void SiPixelResidualModule::book(const edm::ParameterSet& iConfig) {
  DaqMonitorBEInterface* _dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  std::string hisID;
  edm::InputTag src = iConfig.getParameter<edm::InputTag>("src");
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId(src.label());

  hisID = theHistogramId->setHistoId("residualX",id_);
  meResidualX_ = _dbe->book1D(hisID,"Hit-to-Track Residual in X",500,-5.,5.);

  hisID = theHistogramId->setHistoId("residualY",id_);
  meResidualY_ = _dbe->book1D(hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
}


void SiPixelResidualModule::fill(const Measurement2DVector& residual) {
  (meResidualX_)->Fill(residual.x()); 
  (meResidualY_)->Fill(residual.y()); 
}
