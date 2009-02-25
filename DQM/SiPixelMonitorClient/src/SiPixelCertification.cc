#include "DQM/SiPixelMonitorClient/interface/SiPixelCertification.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace std;
using namespace edm;
SiPixelCertification::SiPixelCertification(const edm::ParameterSet& ps) {
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::Creating SiPixelCertification ";
  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
}

SiPixelCertification::~SiPixelCertification(){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::Deleting SiPixelCertification ";
}

void SiPixelCertification::beginJob(const edm::EventSetup& iSetup){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::beginJob ";

  dbe_->setCurrentFolder("Pixel/EventInfo/CertificationContents");

  CertificationPixel= dbe_->bookFloat("PixelFraction");  
  CertificationBarrel= dbe_->bookFloat("PixelBarrelFraction");  
  CertificationShellmI= dbe_->bookFloat("PixelShellmIFraction");  
  CertificationShellmO= dbe_->bookFloat("PixelShellmOFraction");  
  CertificationShellpI= dbe_->bookFloat("PixelShellpIFraction");  
  CertificationShellpO= dbe_->bookFloat("PixelShellpOFraction");  
  CertificationEndcap= dbe_->bookFloat("PixelEndcapFraction");  
  CertificationHalfCylindermI= dbe_->bookFloat("PixelHalfCylindermIFraction");  
  CertificationHalfCylindermO= dbe_->bookFloat("PixelHalfCylindermOFraction");  
  CertificationHalfCylinderpI= dbe_->bookFloat("PixelHalfCylinderpIFraction");  
  CertificationHalfCylinderpO= dbe_->bookFloat("PixelHalfCylinderpOFraction");  

  CertificationPixel->Fill(-1.);  
  CertificationBarrel->Fill(-1.);  
  CertificationShellmI->Fill(-1.);    
  CertificationShellmO->Fill(-1.);   
  CertificationShellpI->Fill(-1.);   
  CertificationShellpO->Fill(-1.);  
  CertificationEndcap->Fill(-1.);  
  CertificationHalfCylindermI->Fill(-1.);
  CertificationHalfCylindermO->Fill(-1.);
  CertificationHalfCylinderpI->Fill(-1.);
  CertificationHalfCylinderpO->Fill(-1.);
}


void SiPixelCertification::beginLuminosityBlock(const LuminosityBlock& lumiBlock, const  EventSetup& iSetup){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::beginLuminosityBlock ";
}


void SiPixelCertification::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::endLuminosityBlock ";

// Compute and fill overall certification bits, for now use smallest single value:
  float dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelDcsFraction"))->getFloatValue();
  float daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelDaqFraction"))->getFloatValue();
  float dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelDqmFraction"))->getFloatValue();
  float pixel_all = std::min(dcsFrac,daqFrac);
  pixel_all = std::min(pixel_all,dqmFrac);
  CertificationPixel = dbe_->get("Pixel/EventInfo/CertificationContents/PixelFraction");
  if(CertificationPixel) CertificationPixel->Fill(pixel_all);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelBarrelDcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelBarrelDaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelDqmFraction"))->getFloatValue();
  float pixel_barrel = std::min(dcsFrac,daqFrac);
  pixel_barrel = std::min(pixel_barrel,dqmFrac);
  CertificationBarrel = dbe_->get("Pixel/EventInfo/CertificationContents/PixelBarrelFraction");
  if(CertificationBarrel) CertificationBarrel->Fill(pixel_barrel);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelShellmIDcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelShellmIDaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelShellmIDqmFraction"))->getFloatValue();
  float pixel_shellmI = std::min(dcsFrac,daqFrac);
  pixel_shellmI = std::min(pixel_shellmI,dqmFrac);
  CertificationShellmI = dbe_->get("Pixel/EventInfo/CertificationContents/PixelShellmIFraction");
  if(CertificationShellmI) CertificationShellmI->Fill(pixel_shellmI);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelShellmODcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelShellmODaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelShellmODqmFraction"))->getFloatValue();
  float pixel_shellmO = std::min(dcsFrac,daqFrac);
  pixel_shellmO = std::min(pixel_shellmO,dqmFrac);
  CertificationShellmO = dbe_->get("Pixel/EventInfo/CertificationContents/PixelShellmOFraction");
  if(CertificationShellmO) CertificationShellmO->Fill(pixel_shellmO);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelShellpIDcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelShellpIDaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelShellpIDqmFraction"))->getFloatValue();
  float pixel_shellpI = std::min(dcsFrac,daqFrac);
  pixel_shellpI = std::min(pixel_shellpI,dqmFrac);
  CertificationShellpI = dbe_->get("Pixel/EventInfo/CertificationContents/PixelShellpIFraction");
  if(CertificationShellpI) CertificationShellpI->Fill(pixel_shellpI);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelShellpODcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelShellpODaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelShellpODqmFraction"))->getFloatValue();
  float pixel_shellpO = std::min(dcsFrac,daqFrac);
  pixel_shellpO = std::min(pixel_shellpO,dqmFrac);
  CertificationShellpO = dbe_->get("Pixel/EventInfo/CertificationContents/PixelShellpOFraction");
  if(CertificationShellpO) CertificationShellpO->Fill(pixel_shellpO);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelEndcapDcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelEndcapDaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapDqmFraction"))->getFloatValue();
  float pixel_endcap = std::min(dcsFrac,daqFrac);
  pixel_endcap = std::min(pixel_endcap,dqmFrac);
  CertificationEndcap = dbe_->get("Pixel/EventInfo/CertificationContents/PixelEndcapFraction");
  if(CertificationEndcap) CertificationEndcap->Fill(pixel_endcap);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelHalfCylindermIDcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelHalfCylindermIDaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelHalfCylindermIDqmFraction"))->getFloatValue();
  float pixel_halfcylindermI = std::min(dcsFrac,daqFrac);
  pixel_halfcylindermI = std::min(pixel_halfcylindermI,dqmFrac);
  CertificationHalfCylindermI = dbe_->get("Pixel/EventInfo/CertificationContents/PixelHalfCylindermIFraction");
  if(CertificationHalfCylindermI) CertificationHalfCylindermI->Fill(pixel_halfcylindermI);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelHalfCylindermODcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelHalfCylindermODaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelHalfCylindermODqmFraction"))->getFloatValue();
  float pixel_halfcylindermO = std::min(dcsFrac,daqFrac);
  pixel_halfcylindermO = std::min(pixel_halfcylindermO,dqmFrac);
  CertificationHalfCylindermO = dbe_->get("Pixel/EventInfo/CertificationContents/PixelHalfCylindermOFraction");
  if(CertificationHalfCylindermO) CertificationHalfCylindermO->Fill(pixel_halfcylindermO);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelHalfCylinderpIDcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelHalfCylinderpIDaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelHalfCylinderpIDqmFraction"))->getFloatValue();
  float pixel_halfcylinderpI = std::min(dcsFrac,daqFrac);
  pixel_halfcylinderpI = std::min(pixel_halfcylinderpI,dqmFrac);
  CertificationHalfCylinderpI = dbe_->get("Pixel/EventInfo/CertificationContents/PixelHalfCylinderpIFraction");
  if(CertificationHalfCylinderpI) CertificationHalfCylinderpI->Fill(pixel_halfcylinderpI);
  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelHalfCylinderpODcsFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelHalfCylinderpODaqFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelHalfCylinderpODqmFraction"))->getFloatValue();
  float pixel_halfcylinderpO = std::min(dcsFrac,daqFrac);
  pixel_halfcylinderpO = std::min(pixel_halfcylinderpO,dqmFrac);
  CertificationHalfCylinderpO = dbe_->get("Pixel/EventInfo/CertificationContents/PixelHalfCylinderpOFraction");
  if(CertificationHalfCylinderpO) CertificationHalfCylinderpO->Fill(pixel_halfcylinderpO);

}


void SiPixelCertification::endJob() {
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::endJob ";
}



void SiPixelCertification::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::analyze ";
}
