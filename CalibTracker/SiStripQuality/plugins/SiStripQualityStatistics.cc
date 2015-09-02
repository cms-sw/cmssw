// -*- C++ -*-
//
// Package:    SiStripQualityStatistics
// Class:      SiStripQualityStatistics
// 
/**\class SiStripQualityStatistics SiStripQualityStatistics.h CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 12:11:10 CEST 2007
//
//
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <sys/time.h>


SiStripQualityStatistics::SiStripQualityStatistics( const edm::ParameterSet& iConfig ):
  m_cacheID_(0), 
  dataLabel_(iConfig.getUntrackedParameter<std::string>("dataLabel","")),
  TkMapFileName_(iConfig.getUntrackedParameter<std::string>("TkMapFileName","")),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  saveTkHistoMap_(iConfig.getUntrackedParameter<bool>("SaveTkHistoMap",true)),
  runNumberA_(iConfig.getUntrackedParameter<unsigned long long>("RunNumber",0)),
  tkMap(0),tkMapFullIOVs(0)
{  
  reader = new SiStripDetInfoFileReader(fp_.fullPath());

  std::stringstream ssRunNumA_;
  ssRunNumA_ << runNumberA_;
  std::string sRunNumberA = ssRunNumA_.str();
  tkMapFullIOVs=new TrackerMap( "Run: "+sRunNumberA+ ", Fraction of Bad Components per module" );
  tkhisto=0;
  if (TkMapFileName_!=""){
    tkhisto   =new TkHistoMap("BadComp","BadComp",-1.); //here the baseline (the value of the empty,not assigned bins) is put to -1 (default is zero)
  }
}

void SiStripQualityStatistics::endJob(){

  std::string filename=TkMapFileName_;
  if (filename!=""){
    tkMapFullIOVs->save(false,0,0,filename.c_str());
    filename.erase(filename.begin()+filename.find("."),filename.end());
    tkMapFullIOVs->print(false,0,0,filename.c_str());
  
    if(saveTkHistoMap_){
      tkhisto->save(filename+".root");
      tkhisto->saveAsCanvas(filename+"_Canvas.root","E");
    }
  }
}

void SiStripQualityStatistics::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  unsigned long long cacheID = iSetup.get<SiStripQualityRcd>().cacheIdentifier();

  std::stringstream ss;

  if (m_cacheID_ == cacheID) 
    return;

  m_cacheID_ = cacheID; 

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  iSetup.get<SiStripQualityRcd>().get(dataLabel_,SiStripQuality_);
    
  for(int i=0;i<4;++i){
    NTkBadComponent[i]=0;
    for(int j=0;j<19;++j){
      ssV[i][j].str("");
      for(int k=0;k<4;++k)
	NBadComponent[i][j][k]=0;
    }
  }

  if (tkMap)
    delete tkMap;

  std::stringstream ssRunNumA_;
  ssRunNumA_ << runNumberA_;
  std::string sRunNumberA = ssRunNumA_.str();
  tkMap=new TrackerMap( "Run: "+sRunNumberA+ ", Type of Bad Components per module" );

  ss.str(""); 
  std::vector<uint32_t> detids=reader->getAllDetIds();
  std::vector<uint32_t>::const_iterator idet=detids.begin();
  for(;idet!=detids.end();++idet){
    ss << "detid " << (*idet) << " IsModuleUsable " << SiStripQuality_->IsModuleUsable((*idet)) << "\n";
    if (SiStripQuality_->IsModuleUsable((*idet)))
      tkMap->fillc(*idet,0x00ff00);
  }
  LogDebug("SiStripQualityStatistics") << ss.str() << std::endl;


  std::vector<SiStripQuality::BadComponent> BC = SiStripQuality_->getBadComponentList();
  
  for (size_t i=0;i<BC.size();++i){
    
    //&&&&&&&&&&&&&
    //Full Tk
    //&&&&&&&&&&&&&

    if (BC[i].BadModule) 
      NTkBadComponent[0]++;
    if (BC[i].BadFibers) 
      NTkBadComponent[1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
    if (BC[i].BadApvs)
      NTkBadComponent[2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );

    //&&&&&&&&&&&&&&&&&
    //Single SubSyste
    //&&&&&&&&&&&&&&&&&
    int component;
    DetId detectorId=DetId(BC[i].detid);
    int subDet = detectorId.subdetId();
    if ( subDet == StripSubdetector::TIB ){
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&
      
      component=tTopo->tibLayer(BC[i].detid);
      SetBadComponents(0, component, BC[i]);         

    } else if ( subDet == StripSubdetector::TID ) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&

      component=tTopo->tidSide(BC[i].detid)==2?tTopo->tidWheel(BC[i].detid):tTopo->tidWheel(BC[i].detid)+3;
      SetBadComponents(1, component, BC[i]);         

    } else if ( subDet == StripSubdetector::TOB ) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&

      component=tTopo->tobLayer(BC[i].detid);
      SetBadComponents(2, component, BC[i]);         

    } else if ( subDet == StripSubdetector::TEC ) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&

      component=tTopo->tecSide(BC[i].detid)==2?tTopo->tecWheel(BC[i].detid):tTopo->tecWheel(BC[i].detid)+9;
      SetBadComponents(3, component, BC[i]);         

    }    
  }

  //&&&&&&&&&&&&&&&&&&
  // Single Strip Info
  //&&&&&&&&&&&&&&&&&&
  float percentage=0;

  SiStripQuality::RegistryIterator rbegin = SiStripQuality_->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend   = SiStripQuality_->getRegistryVectorEnd();
  
  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
    uint32_t detid=rp->detid;

    int subdet=-999; int component=-999;
    DetId detectorId=DetId(detid);
    int subDet = detectorId.subdetId();
    if ( subDet == StripSubdetector::TIB ){
      subdet=0;
      component=tTopo->tibLayer(detid);
    } else if ( subDet == StripSubdetector::TID ) {
      subdet=1;
      component=tTopo->tidSide(detid)==2?tTopo->tidWheel(detid):tTopo->tidWheel(detid)+3;
    } else if ( subDet == StripSubdetector::TOB ) {
      subdet=2;
      component=tTopo->tobLayer(detid);
    } else if ( subDet == StripSubdetector::TEC ) {
      subdet=3;
      component=tTopo->tecSide(detid)==2?tTopo->tecWheel(detid):tTopo->tecWheel(detid)+9;
    } 

    SiStripQuality::Range sqrange = SiStripQuality::Range( SiStripQuality_->getDataVectorBegin()+rp->ibegin , SiStripQuality_->getDataVectorBegin()+rp->iend );
        
    percentage=0;
    for(int it=0;it<sqrange.second-sqrange.first;it++){
      unsigned int range=SiStripQuality_->decode( *(sqrange.first+it) ).range;
      NTkBadComponent[3]+=range;
      NBadComponent[subdet][0][3]+=range;
      NBadComponent[subdet][component][3]+=range;
      percentage+=range;
    }
    if(percentage!=0)
      percentage/=128.*reader->getNumberOfApvsAndStripLength(detid).first;
    if(percentage>1)
      edm::LogError("SiStripQualityStatistics") <<  "PROBLEM detid " << detid << " value " << percentage<< std::endl;

    //------- Global Statistics on percentage of bad components along the IOVs ------//
    tkMapFullIOVs->fill(detid,percentage);
    if(tkhisto!=NULL)
      tkhisto->fill(detid,percentage);
  }
  
 
  //&&&&&&&&&&&&&&&&&&
  // printout
  //&&&&&&&&&&&&&&&&&&

  ss.str("");
  ss << "\n-----------------\nNew IOV starting from run " <<   e.id().run() << " event " << e.id().event() << " lumiBlock " << e.luminosityBlock() << " time " << e.time().value() << " chacheID " << m_cacheID_ << "\n-----------------\n";
  ss << "\n-----------------\nGlobal Info\n-----------------";
  ss << "\nBadComponent \t   Modules \tFibers \tApvs\tStrips\n----------------------------------------------------------------";
  ss << "\nTracker:\t\t"<<NTkBadComponent[0]<<"\t"<<NTkBadComponent[1]<<"\t"<<NTkBadComponent[2]<<"\t"<<NTkBadComponent[3];
  ss<< "\n";
  ss << "\nTIB:\t\t\t"<<NBadComponent[0][0][0]<<"\t"<<NBadComponent[0][0][1]<<"\t"<<NBadComponent[0][0][2]<<"\t"<<NBadComponent[0][0][3];
  ss << "\nTID:\t\t\t"<<NBadComponent[1][0][0]<<"\t"<<NBadComponent[1][0][1]<<"\t"<<NBadComponent[1][0][2]<<"\t"<<NBadComponent[1][0][3];
  ss << "\nTOB:\t\t\t"<<NBadComponent[2][0][0]<<"\t"<<NBadComponent[2][0][1]<<"\t"<<NBadComponent[2][0][2]<<"\t"<<NBadComponent[2][0][3];
  ss << "\nTEC:\t\t\t"<<NBadComponent[3][0][0]<<"\t"<<NBadComponent[3][0][1]<<"\t"<<NBadComponent[3][0][2]<<"\t"<<NBadComponent[3][0][3];
  ss << "\n";

  for (int i=1;i<5;++i)
    ss << "\nTIB Layer " << i   << " :\t\t"<<NBadComponent[0][i][0]<<"\t"<<NBadComponent[0][i][1]<<"\t"<<NBadComponent[0][i][2]<<"\t"<<NBadComponent[0][i][3];
  ss << "\n";
  for (int i=1;i<4;++i)
    ss << "\nTID+ Disk " << i   << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
  for (int i=4;i<7;++i)
    ss << "\nTID- Disk " << i-3 << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
  ss << "\n";
  for (int i=1;i<7;++i)
    ss << "\nTOB Layer " << i   << " :\t\t"<<NBadComponent[2][i][0]<<"\t"<<NBadComponent[2][i][1]<<"\t"<<NBadComponent[2][i][2]<<"\t"<<NBadComponent[2][i][3];
  ss << "\n";
  for (int i=1;i<10;++i)
    ss << "\nTEC+ Disk " << i   << " :\t\t"<<NBadComponent[3][i][0]<<"\t"<<NBadComponent[3][i][1]<<"\t"<<NBadComponent[3][i][2]<<"\t"<<NBadComponent[3][i][3];
  for (int i=10;i<19;++i)
    ss << "\nTEC- Disk " << i-9 << " :\t\t"<<NBadComponent[3][i][0]<<"\t"<<NBadComponent[3][i][1]<<"\t"<<NBadComponent[3][i][2]<<"\t"<<NBadComponent[3][i][3];
  ss<< "\n";

  ss << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers Apvs\n----------------------------------------------------------------";
  for (int i=1;i<5;++i)
    ss << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  ss << "\n";
  for (int i=1;i<4;++i)
    ss << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i=4;i<7;++i)
    ss << "\nTID- Disk " << i-3 << " :" << ssV[1][i].str();
  ss << "\n";
  for (int i=1;i<7;++i)
    ss << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  ss << "\n";
  for (int i=1;i<10;++i)
    ss << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i=10;i<19;++i)
    ss << "\nTEC- Disk " << i-9 << " :" << ssV[3][i].str();


  edm::LogInfo("SiStripQualityStatistics") << ss.str() << std::endl;

  std::string filetype=TkMapFileName_,filename=TkMapFileName_;
  std::stringstream sRun; sRun.str("");
  sRun << "_Run_"  << std::setw(6) << std::setfill('0')<< e.id().run() << std::setw(0) ;
    
  if (filename!=""){
    filename.insert(filename.find("."),sRun.str());
    tkMap->save(true,0,0,filename.c_str());
    filename.erase(filename.begin()+filename.find("."),filename.end());
    tkMap->print(true,0,0,filename.c_str());
  }
}


void SiStripQualityStatistics::SetBadComponents(int i, int component,SiStripQuality::BadComponent& BC){

  int napv=reader->getNumberOfApvsAndStripLength(BC.detid).first;

  ssV[i][component] << "\n\t\t " 
		    << BC.detid 
		    << " \t " << BC.BadModule << " \t " 
		    << ( (BC.BadFibers)&0x1 ) << " ";
  if (napv==4)
    ssV[i][component] << "x " <<( (BC.BadFibers>>1)&0x1 );
  
  if (napv==6)
    ssV[i][component] << ( (BC.BadFibers>>1)&0x1 ) << " "
		      << ( (BC.BadFibers>>2)&0x1 );
  ssV[i][component] << " \t " 
		    << ( (BC.BadApvs)&0x1 ) << " " 
		    << ( (BC.BadApvs>>1)&0x1 ) << " ";
  if (napv==4) 
    ssV[i][component] << "x x " << ( (BC.BadApvs>>2)&0x1 ) << " " 
		      << ( (BC.BadApvs>>3)&0x1 );
  if (napv==6) 
    ssV[i][component] << ( (BC.BadApvs>>2)&0x1 ) << " " 
		      << ( (BC.BadApvs>>3)&0x1 ) << " " 
		      << ( (BC.BadApvs>>4)&0x1 ) << " " 
		      << ( (BC.BadApvs>>5)&0x1 ) << " "; 

  if (BC.BadApvs){
    NBadComponent[i][0][2]+= ( (BC.BadApvs>>5)&0x1 )+ ( (BC.BadApvs>>4)&0x1 ) + ( (BC.BadApvs>>3)&0x1 ) + 
      ( (BC.BadApvs>>2)&0x1 )+ ( (BC.BadApvs>>1)&0x1 ) + ( (BC.BadApvs)&0x1 );
    NBadComponent[i][component][2]+= ( (BC.BadApvs>>5)&0x1 )+ ( (BC.BadApvs>>4)&0x1 ) + ( (BC.BadApvs>>3)&0x1 ) + 
      ( (BC.BadApvs>>2)&0x1 )+ ( (BC.BadApvs>>1)&0x1 ) + ( (BC.BadApvs)&0x1 );
    tkMap->fillc(BC.detid,0xff0000);
  }
  if (BC.BadFibers){ 
    NBadComponent[i][0][1]+= ( (BC.BadFibers>>2)&0x1 )+ ( (BC.BadFibers>>1)&0x1 ) + ( (BC.BadFibers)&0x1 );
    NBadComponent[i][component][1]+= ( (BC.BadFibers>>2)&0x1 )+ ( (BC.BadFibers>>1)&0x1 ) + ( (BC.BadFibers)&0x1 );
    tkMap->fillc(BC.detid,0x0000ff);
  }   
  if (BC.BadModule){
    NBadComponent[i][0][0]++;
    NBadComponent[i][component][0]++;
    tkMap->fillc(BC.detid,0x0);
  }
}
