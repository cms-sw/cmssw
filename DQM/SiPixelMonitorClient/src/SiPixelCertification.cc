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

  dbe_->setCurrentFolder("Pixel/EventInfo");
  CertificationPixel= dbe_->bookFloat("CertificationSummary");  
  dbe_->setCurrentFolder("Pixel/EventInfo/CertificationContents");
  CertificationBarrel= dbe_->bookFloat("PixelBarrelFraction");  
  CertificationEndcap= dbe_->bookFloat("PixelEndcapFraction");  

  CertificationPixel->Fill(1.);  
  CertificationBarrel->Fill(1.);  
  CertificationEndcap->Fill(1.);  
}


void SiPixelCertification::beginLuminosityBlock(const LuminosityBlock& lumiBlock, const  EventSetup& iSetup){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::beginLuminosityBlock ";
}


void SiPixelCertification::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){
//cout<<"Entering SiPixelCertification::endLuminosityBlock: "<<endl;
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::endLuminosityBlock ";
  // Compute and fill overall certification bits, for now use smallest single value:
  float dcsFrac = (dbe_->get("Pixel/EventInfo/DCSSummary"))->getFloatValue();
  float daqFrac = (dbe_->get("Pixel/EventInfo/DAQSummary"))->getFloatValue();
  float dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummary"))->getFloatValue();
  float pixel_all = std::min(dcsFrac,daqFrac);
  pixel_all = std::min(pixel_all,dqmFrac);
//std::cout<<"Pixel numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_all<<std::endl;
  CertificationPixel = dbe_->get("Pixel/EventInfo/CertificationSummary");
  if(CertificationPixel) CertificationPixel->Fill(pixel_all);

  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelBarrelFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelBarrelFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction"))->getFloatValue();
  float pixel_barrel = std::min(dcsFrac,daqFrac);
  pixel_barrel = std::min(pixel_barrel,dqmFrac);
//std::cout<<"Barrel numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_barrel<<std::endl;
  CertificationBarrel = dbe_->get("Pixel/EventInfo/CertificationContents/PixelBarrelFraction");
  if(CertificationBarrel) CertificationBarrel->Fill(pixel_barrel);

  dcsFrac = (dbe_->get("Pixel/EventInfo/DCSContents/PixelEndcapFraction"))->getFloatValue();
  daqFrac = (dbe_->get("Pixel/EventInfo/DAQContents/PixelEndcapFraction"))->getFloatValue();
  dqmFrac = (dbe_->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction"))->getFloatValue();
  float pixel_endcap = std::min(dcsFrac,daqFrac);
  pixel_endcap = std::min(pixel_endcap,dqmFrac);
//std::cout<<"Endcap numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_endcap<<std::endl;
  CertificationEndcap = dbe_->get("Pixel/EventInfo/CertificationContents/PixelEndcapFraction");
  if(CertificationEndcap) CertificationEndcap->Fill(pixel_endcap);

}


void SiPixelCertification::endJob() {
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::endJob ";
}



void SiPixelCertification::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::analyze ";
}
