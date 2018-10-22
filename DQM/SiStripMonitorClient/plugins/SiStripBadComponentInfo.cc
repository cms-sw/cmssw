#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/plugins/SiStripBadComponentInfo.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

//
// -- Contructor
//
SiStripBadComponentInfo::SiStripBadComponentInfo(edm::ParameterSet const& pSet) : 
    bookedStatus_(false),
    nSubSystem_(6),
    qualityLabel_(pSet.getParameter<std::string>("StripQualityLabel"))
{ 
  // Create MessageSender
  LogDebug( "SiStripBadComponentInfo") << "SiStripBadComponentInfo::Deleting SiStripBadComponentInfo ";
}
//
// -- Destructor
//
SiStripBadComponentInfo::~SiStripBadComponentInfo() {
  LogDebug("SiStripBadComponentInfo") << "SiStripBadComponentInfo::Deleting SiStripBadComponentInfo ";
  mapBadAPV.clear();
  mapBadFiber.clear();
  mapBadStrip.clear();
}
//
// -- Begin Run
//
void SiStripBadComponentInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
}
//
// -- Read Condition
//
void SiStripBadComponentInfo::checkBadComponents(edm::EventSetup const& eSetup) {

  LogDebug ("SiStripBadComponentInfo") <<"SiStripBadComponentInfo:: Begining of Run";

  //Retrieve tracker topology from geometry
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* const topo = tTopoHandle_.product();
 
  eSetup.get<SiStripQualityRcd>().get(qualityLabel_,siStripQuality_);
 
  auto const& BC = siStripQuality_->getBadComponentList();
  
  for (size_t i=0;i<BC.size();++i){
    int subdet=-999; int component=-999;
    //&&&&&&&&&&&&&&&&&
    //Single SubSyste
    //&&&&&&&&&&&&&&&&&
    int subDet = DetId(BC[i].detid).subdetId();
    if ( subDet == StripSubdetector::TIB ){
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&
      subdet = 3;
      component=topo->tibLayer(BC[i].detid);
    } else if ( subDet == StripSubdetector::TID ) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&
      if (topo->tidSide(BC[i].detid)==2) subdet = 4;
      else subdet = 5;
      component = topo->tidWheel(BC[i].detid);
    } else if ( subDet == StripSubdetector::TOB ) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&
      subdet = 6;
      component=topo->tobLayer(BC[i].detid);
    } else if ( subDet == StripSubdetector::TEC ) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&
      if (topo->tecSide(BC[i].detid)==2) subdet = 2;
      else  subdet=1;
      component=topo->tecWheel(BC[i].detid);
    }
    fillBadComponentMaps(subdet,component,BC[i]);        
  }

  //&&&&&&&&&&&&&&&&&&
  // Single Strip Info
  //&&&&&&&&&&&&&&&&&&

  SiStripQuality::RegistryIterator rbegin = siStripQuality_->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend   = siStripQuality_->getRegistryVectorEnd();
  
  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
    uint32_t detid=rp->detid;   
    int subdet=-999; int component=-999;
    DetId detectorId=DetId(detid);
    int subDet = detectorId.subdetId();
    if ( subDet == StripSubdetector::TIB ){
      subdet=3;
      component=topo->tibLayer(detid);
    } else if ( subDet == StripSubdetector::TID ) {
      if (topo->tidSide(detid)==2) subdet = 5;
      else subdet = 4;
      component = topo->tidWheel(detid);
    } else if ( subDet == StripSubdetector::TOB ) {
      subdet=6;
      component=topo->tobLayer(detid);
    } else if ( subDet == StripSubdetector::TEC ) {
      if (topo->tecSide(detid) == 2) subdet = 2;
      else  subdet=1;
      component=topo->tecWheel(detid);
    } 

    SiStripQuality::Range sqrange = SiStripQuality::Range( siStripQuality_->getDataVectorBegin()+rp->ibegin , siStripQuality_->getDataVectorBegin()+rp->iend );
        
    for(int it=0;it<sqrange.second-sqrange.first;it++){
      unsigned int range=siStripQuality_->decode( *(sqrange.first+it) ).range;
      float val = (mapBadStrip.find(std::make_pair(subdet,component))!=mapBadStrip.end()) ? mapBadStrip.at(std::make_pair(subdet,component)) : 0.; 
      val += range;
      mapBadStrip[std::make_pair(subdet,component)]=val;
    }
  }
}
//
// -- End Run
//
void SiStripBadComponentInfo::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  LogDebug ("SiStripBadComponentInfo") <<"SiStripBadComponentInfo:: End of Run";
  checkBadComponents(eSetup);  
}
//
// -- End Job
//
void SiStripBadComponentInfo::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter){
  LogDebug ("SiStripBadComponentInfo") <<"SiStripBadComponentInfo::dqmEndRun";
  bookBadComponentHistos(ibooker, igetter);
  createSummary(badAPVME_  ,mapBadAPV  ); 
  createSummary(badFiberME_,mapBadFiber); 
  createSummary(badStripME_,mapBadStrip); 
} 
//
// -- Book MEs for SiStrip Dcs Fraction
//
void SiStripBadComponentInfo::bookBadComponentHistos(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  if (!bookedStatus_) {
    std::string strip_dir = "";
    ibooker.cd();
    //    SiStripUtility::getTopFolderPath(dqmStore_, "SiStrip", strip_dir); 
    if (igetter.dirExists("SiStrip")) {
      ibooker.cd("SiStrip");
      strip_dir = ibooker.pwd();
    }
    edm::LogInfo ("SiStripBadComponentInfo") << "SiStripBadComponentInfo::bookBadComponentHistos ==> " << strip_dir << " " << ibooker.pwd() << std::endl;
    if (!strip_dir.empty()) ibooker.setCurrentFolder(strip_dir+"/EventInfo");
    else ibooker.setCurrentFolder("SiStrip/EventInfo");
       
    ibooker.cd();
    if (!strip_dir.empty())  ibooker.setCurrentFolder(strip_dir+"/EventInfo/BadComponentContents");

    std::string  hname, htitle;
    hname  = "BadAPVMap";
    htitle = "SiStrip Bad APVs";
    badAPVME_ = ibooker.book2D(hname, htitle, nSubSystem_, 0.5, nSubSystem_+0.5, 10, 0.5, 10.5);
    badAPVME_->setAxisTitle("Sub Detector Type", 1);
    badAPVME_->setAxisTitle("Layer/Disc Number", 2);

    hname  = "BadFiberMap";
    htitle = "SiStrip Bad Fibers";
    badFiberME_ = ibooker.book2D(hname, htitle, nSubSystem_, 0.5, nSubSystem_+0.5, 10, 0.5, 10.5);
    badFiberME_->setAxisTitle("Sub Detector Type", 1);
    badFiberME_->setAxisTitle("Layer/Disc Number", 2);

    hname  = "BadStripMap";
    htitle = "SiStrip Bad Strips";
    badStripME_ = ibooker.book2D(hname, htitle, nSubSystem_, 0.5, nSubSystem_+0.5, 10, 0.5, 10.5);
    badStripME_->setAxisTitle("Sub Detector Type", 1);
    badStripME_->setAxisTitle("Layer/Disc Number", 2);

    std::vector<std::string> names;
    names.push_back("TECB");
    names.push_back("TECF");
    names.push_back("TIB");
    names.push_back("TIDB");
    names.push_back("TIDF");
    names.push_back("TOB");

    for (unsigned int i=0; i < names.size(); i++) {
      badAPVME_->setBinLabel(i+1, names[i]);       
      badFiberME_->setBinLabel(i+1, names[i]);       
      badStripME_->setBinLabel(i+1, names[i]);       
    }

    bookedStatus_ = true;
    ibooker.cd();
  }
}
void SiStripBadComponentInfo::fillBadComponentMaps(int xbin,int component,SiStripQuality::BadComponent const& BC){

  auto index = std::make_pair(xbin,component);

  if (BC.BadApvs){ 
    int ntot =  std::bitset<16>(BC.BadApvs&0x3f).count();    
    float val = (mapBadAPV.find(index)!=mapBadAPV.end()) ? mapBadAPV.at(index) : 0.; 
    val += ntot;
    mapBadAPV[index]=val;
  }
  if (BC.BadFibers){ 
    int ntot = std::bitset<16>(BC.BadFibers&0x7).count();
    float val = (mapBadFiber.find(index)!=mapBadFiber.end()) ? mapBadFiber.at(index) : 0.; 
    val+= ntot;
    mapBadFiber[index]=val;
  }   
}
void SiStripBadComponentInfo::createSummary(MonitorElement* me,const std::map<std::pair<int,int>,float >& map) {
  for (int i=1; i<nSubSystem_+1; i++) {
    float sum = 0.0;
    for (int k=1; k<me->getNbinsY(); k++) {
      auto index = std::make_pair(i,k);
      if (map.find(index)!=map.end()){
	me->setBinContent(i,k,map.at(index)); // fill the layer/wheel bins
	sum += map.at(index);      
      }
    }
    me->setBinContent(i,me->getNbinsY(), sum); // fill the summary bin (last one)
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripBadComponentInfo);
