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
// $Id: SiStripQualityStatistics.cc,v 1.7 2007/11/12 16:05:01 giordano Exp $
//
//
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <sys/time.h>


SiStripQualityStatistics::SiStripQualityStatistics( const edm::ParameterSet& iConfig ):
  m_cacheID_(0), 
  dataLabel_(iConfig.getUntrackedParameter<std::string>("dataLabel","")),
  TkMapFileName_(iConfig.getUntrackedParameter<std::string>("TkMapFileName","TkMapBadComponents.svg")),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  tkMap(0)
{  
  reader = new SiStripDetInfoFileReader(fp_.fullPath());
}

void SiStripQualityStatistics::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

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
  tkMap=new TrackerMap( "BadComponents" );

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
    SiStripDetId a(BC[i].detid);
    if ( a.subdetId() == SiStripDetId::TIB ){
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&
      
      component=TIBDetId(BC[i].detid).layer();
      SetBadComponents(0, component, BC[i]);         

    } else if ( a.subdetId() == SiStripDetId::TID ) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&

      component=TIDDetId(BC[i].detid).side()==2?TIDDetId(BC[i].detid).wheel():TIDDetId(BC[i].detid).wheel()+3;
      SetBadComponents(1, component, BC[i]);         

    } else if ( a.subdetId() == SiStripDetId::TOB ) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&

      component=TOBDetId(BC[i].detid).layer();
      SetBadComponents(2, component, BC[i]);         

    } else if ( a.subdetId() == SiStripDetId::TEC ) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&

      component=TECDetId(BC[i].detid).side()==2?TECDetId(BC[i].detid).wheel():TECDetId(BC[i].detid).wheel()+9;
      SetBadComponents(3, component, BC[i]);         

    }    
  }

  //&&&&&&&&&&&&&&&&&&
  // Single Strip Info
  //&&&&&&&&&&&&&&&&&&

  SiStripQuality::RegistryIterator rbegin = SiStripQuality_->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend   = SiStripQuality_->getRegistryVectorEnd();
  
  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
    uint32_t detid=rp->detid;

    int subdet,component;
    SiStripDetId a(detid);
    if ( a.subdetId() == 3 ){
      subdet=0;
      component=TIBDetId(detid).layer();
    } else if ( a.subdetId() == 4 ) {
      subdet=1;
      component=TIDDetId(detid).side()==2?TIDDetId(detid).wheel():TIDDetId(detid).wheel()+3;
    } else if ( a.subdetId() == 5 ) {
      subdet=2;
      component=TOBDetId(detid).layer();
    } else if ( a.subdetId() == 6 ) {
      subdet=3;
      component=TECDetId(detid).side()==2?TECDetId(detid).wheel():TECDetId(detid).wheel()+9;
    } 

    SiStripQuality::Range sqrange = SiStripQuality::Range( SiStripQuality_->getDataVectorBegin()+rp->ibegin , SiStripQuality_->getDataVectorBegin()+rp->iend );
        
    for(int it=0;it<sqrange.second-sqrange.first;it++){
      unsigned int range=SiStripQuality_->decode( *(sqrange.first+it) ).range;
      NTkBadComponent[3]+=range;
      NBadComponent[subdet][0][3]+=range;
      NBadComponent[subdet][component][3]+=range;
    }
  }
  
 
  //&&&&&&&&&&&&&&&&&&
  // printout
  //&&&&&&&&&&&&&&&&&&

  ss.str("");
  ss << "\n-----------------\nNew IOV starting from run " <<   e.id().run() << "\n-----------------\n";
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
  
  filename.insert(filename.find("."),sRun.str());
  tkMap->save(true,0,0,filename.c_str());
  filename.erase(filename.begin()+filename.find("."),filename.end());
  tkMap->print(true,0,0,filename.c_str());
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
