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
  firstLumi = true;
}

SiPixelCertification::~SiPixelCertification(){
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::Deleting SiPixelCertification ";
}

void SiPixelCertification::dqmEndLuminosityBlock(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){
//cout<<"Entering SiPixelCertification::endLuminosityBlock: "<<endl;

  //If first run, book some histograms
  if (firstLumi){
    iBooker.setCurrentFolder("Pixel/EventInfo");
    CertificationPixel= iBooker.bookFloat("CertificationSummary");  
    iBooker.setCurrentFolder("Pixel/EventInfo/CertificationContents");
    CertificationBarrel= iBooker.bookFloat("PixelBarrelFraction");  
    CertificationEndcap= iBooker.bookFloat("PixelEndcapFraction");  

    CertificationPixel->Fill(1.);  
    CertificationBarrel->Fill(1.);  
    CertificationEndcap->Fill(1.);  
    
    firstLumi = false;
  }

  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::endLuminosityBlock ";
  // Compute and fill overall certification bits, for now use smallest single value:
  float dcsFrac = (iGetter.getElement("Pixel/EventInfo/DCSSummary"))->getFloatValue();
  float daqFrac = (iGetter.getElement("Pixel/EventInfo/DAQSummary"))->getFloatValue();
  float dqmFrac = (iGetter.getElement("Pixel/EventInfo/reportSummary"))->getFloatValue();
  float pixel_all = std::min(dcsFrac,daqFrac);
  pixel_all = std::min(pixel_all,dqmFrac);
//std::cout<<"Pixel numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_all<<std::endl;
  CertificationPixel = iGetter.getElement("Pixel/EventInfo/CertificationSummary");
  if(CertificationPixel) CertificationPixel->Fill(pixel_all);

  dcsFrac = (iGetter.getElement("Pixel/EventInfo/DCSContents/PixelBarrelFraction"))->getFloatValue();
  daqFrac = (iGetter.getElement("Pixel/EventInfo/DAQContents/PixelBarrelFraction"))->getFloatValue();
  dqmFrac = (iGetter.getElement("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction"))->getFloatValue();
  float pixel_barrel = std::min(dcsFrac,daqFrac);
  pixel_barrel = std::min(pixel_barrel,dqmFrac);
//std::cout<<"Barrel numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_barrel<<std::endl;
  CertificationBarrel = iGetter.getElement("Pixel/EventInfo/CertificationContents/PixelBarrelFraction");
  if(CertificationBarrel) CertificationBarrel->Fill(pixel_barrel);

  dcsFrac = (iGetter.getElement("Pixel/EventInfo/DCSContents/PixelEndcapFraction"))->getFloatValue();
  daqFrac = (iGetter.getElement("Pixel/EventInfo/DAQContents/PixelEndcapFraction"))->getFloatValue();
  dqmFrac = (iGetter.getElement("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction"))->getFloatValue();
  float pixel_endcap = std::min(dcsFrac,daqFrac);
  pixel_endcap = std::min(pixel_endcap,dqmFrac);
//std::cout<<"Endcap numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_endcap<<std::endl;
  CertificationEndcap = iGetter.getElement("Pixel/EventInfo/CertificationContents/PixelEndcapFraction");
  if(CertificationEndcap) CertificationEndcap->Fill(pixel_endcap);

}


void SiPixelCertification::dqmEndJob(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter){
//cout<<"Entering SiPixelCertification::endRun: "<<endl;
  edm::LogInfo( "SiPixelCertification") << "SiPixelCertification::endRun ";
  // Compute and fill overall certification bits, for now use smallest single value:
  float dcsFrac = (iGetter.getElement("Pixel/EventInfo/DCSSummary"))->getFloatValue();
  float daqFrac = (iGetter.getElement("Pixel/EventInfo/DAQSummary"))->getFloatValue();
  float dqmFrac = (iGetter.getElement("Pixel/EventInfo/reportSummary"))->getFloatValue();
  float pixel_all = std::min(dcsFrac,daqFrac);
  pixel_all = std::min(pixel_all,dqmFrac);
//std::cout<<"Pixel numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_all<<std::endl;
  if(CertificationPixel) CertificationPixel->Fill(pixel_all);

  dcsFrac = (iGetter.getElement("Pixel/EventInfo/DCSContents/PixelBarrelFraction"))->getFloatValue();
  daqFrac = (iGetter.getElement("Pixel/EventInfo/DAQContents/PixelBarrelFraction"))->getFloatValue();
  dqmFrac = (iGetter.getElement("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction"))->getFloatValue();
  float pixel_barrel = std::min(dcsFrac,daqFrac);
  pixel_barrel = std::min(pixel_barrel,dqmFrac);
//std::cout<<"Barrel numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_barrel<<std::endl;
  if(CertificationBarrel) CertificationBarrel->Fill(pixel_barrel);

  dcsFrac = (iGetter.getElement("Pixel/EventInfo/DCSContents/PixelEndcapFraction"))->getFloatValue();
  daqFrac = (iGetter.getElement("Pixel/EventInfo/DAQContents/PixelEndcapFraction"))->getFloatValue();
  dqmFrac = (iGetter.getElement("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction"))->getFloatValue();
  float pixel_endcap = std::min(dcsFrac,daqFrac);
  pixel_endcap = std::min(pixel_endcap,dqmFrac);
//std::cout<<"Endcap numbers: "<<dcsFrac<<" , "<<daqFrac<<" , "<<dqmFrac<<" , "<<pixel_endcap<<std::endl;
  if(CertificationEndcap) CertificationEndcap->Fill(pixel_endcap);

}




