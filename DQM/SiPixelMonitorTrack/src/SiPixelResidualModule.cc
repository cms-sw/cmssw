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
// $Id: SiPixelResidualModule.cc,v 1.3 2007/07/16 23:00:16 schuang Exp $


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
  DaqMonitorBEInterface* dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  edm::InputTag src = iConfig.getParameter<edm::InputTag>("src");
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId(src.label());
  std::string hisID;

  hisID = theHistogramId->setHistoId("residualX",id_);
  meResidualX_ = dbe->book1D(hisID,"Hit-to-Track Residual in X",500,-5.,5.);
  meResidualX_->setAxisTitle("hit-to-track residual in x (cm)",1);

  hisID = theHistogramId->setHistoId("residualY",id_);
  meResidualY_ = dbe->book1D(hisID,"Hit-to-Track Residual in Y",500,-5.,5.);
  meResidualY_->setAxisTitle("hit-to-track residual in y (cm)",1);

  delete theHistogramId;
}


void SiPixelResidualModule::fill(const Measurement2DVector& residual) {
  (meResidualX_)->Fill(residual.x()); 
  (meResidualY_)->Fill(residual.y()); 
}
