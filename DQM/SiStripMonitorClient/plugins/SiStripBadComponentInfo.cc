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
    m_cacheID_(0),
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

}
//
// -- Begin Run
//
void SiStripBadComponentInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  LogDebug ("SiStripBadComponentInfo") <<"SiStripBadComponentInfo:: Begining of Run";

  //Retrieve tracker topology from geometry
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);

  unsigned long long cacheID = eSetup.get<SiStripQualityRcd>().cacheIdentifier();
  if (m_cacheID_ == !cacheID) { 
    
    m_cacheID_ = cacheID; 
    LogDebug("SiStripBadComponentInfo") <<"SiStripBadchannelInfoNew::readCondition : "
				   << " Change in Cache";
    eSetup.get<SiStripQualityRcd>().get(qualityLabel_,siStripQuality_);

  }    
}
//
// -- Read Condition
//
void SiStripBadComponentInfo::checkBadComponents() {
  const TrackerTopology* const topo = tTopoHandle_.product();
  
  std::vector<SiStripQuality::BadComponent> BC = siStripQuality_->getBadComponentList();
  
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
    fillBadComponentHistos(subdet,component,BC[i]);        
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
      float val = badStripME_->getBinContent(subdet, component);
      val += range;
      badStripME_->setBinContent(subdet, component,val);
    }
  }
}
//
// -- End Run
//
void SiStripBadComponentInfo::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter){
  LogDebug ("SiStripBadComponentInfo") <<"SiStripBadComponentInfo::dqmEndRun";
  bookBadComponentHistos(ibooker, igetter);
  checkBadComponents();  
  createSummary(badAPVME_); 
  createSummary(badFiberME_); 
  createSummary(badStripME_); 
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
    std::cout << "SiStripBadComponentInfo::bookBadComponentHistos ==> " << strip_dir << " " << ibooker.pwd() << std::endl;
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
void SiStripBadComponentInfo::fillBadComponentHistos(int xbin,int component,SiStripQuality::BadComponent& BC){

  if (BC.BadApvs){ 
    int ntot =  std::bitset<16>(BC.BadApvs&0x3f).count();
    float val = badAPVME_->getBinContent(xbin, component);
    val += ntot;
    badAPVME_->setBinContent(xbin, component, val);
  }
  if (BC.BadFibers){ 
    int ntot = std::bitset<16>(BC.BadFibers&0x7).count();
    float val = badFiberME_->getBinContent(xbin, component);
    val+= ntot;
    badFiberME_->setBinContent(xbin, component, val);
  }   
}
void SiStripBadComponentInfo::createSummary(MonitorElement* me) {
  for (int i=1; i<nSubSystem_+1; i++) {
   float sum = 0.0;
   for (int k=1; k<me->getNbinsY(); k++) {
     if (me->getBinContent(i,k)) sum+= me->getBinContent(i,k);      
   }
   me->setBinContent(i,me->getNbinsY(), sum);
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripBadComponentInfo);
