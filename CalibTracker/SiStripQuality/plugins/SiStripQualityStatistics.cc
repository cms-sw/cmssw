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
// $Id: SiStripQualityStatistics.cc,v 1.2 2007/10/11 12:54:14 giordano Exp $
//
//
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <sstream>

SiStripQualityStatistics::SiStripQualityStatistics( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)),
  m_cacheID_(0){}

void SiStripQualityStatistics::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  unsigned long long cacheID = iSetup.get<SiStripQualityRcd>().cacheIdentifier();

  std::stringstream ss;

  if (m_cacheID_ == cacheID) 
    return;

  m_cacheID_ = cacheID; 

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  iSetup.get<SiStripQualityRcd>().get(SiStripQuality_);
  
  //Global Info
  int NTkBadComponent[4]; //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  int NBadComponent[4][19][4];  
  //legend: NBadComponent[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
  //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
  //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips

  for(int i=0;i<4;++i){
    NTkBadComponent[i]=0;
    for(int j=0;j<19;++j)
      for(int k=0;k<4;++k)
	NBadComponent[i][j][k]=0;
  }

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
    if ( a.subdetId() == 3 ){
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&
      
      component=TIBDetId(BC[i].detid).layer();
  
      if (BC[i].BadModule){
	NBadComponent[0][0][0]++;
	NBadComponent[0][component][0]++;
      }
      if (BC[i].BadFibers){ 
	NBadComponent[0][0][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
	NBadComponent[0][component][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
      }
      if (BC[i].BadApvs){
	NBadComponent[0][0][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
	NBadComponent[0][component][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
      }

    } else if ( a.subdetId() == 4 ) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&

      component=TIDDetId(BC[i].detid).side()==2?TIDDetId(BC[i].detid).wheel()+1:TIDDetId(BC[i].detid).wheel()+4;
         
      if (BC[i].BadModule){
	NBadComponent[1][0][0]++;
	NBadComponent[1][component][0]++;
      }
      if (BC[i].BadFibers){ 
	NBadComponent[1][0][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
	NBadComponent[1][component][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
      }
      if (BC[i].BadApvs){
	NBadComponent[1][0][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
	NBadComponent[1][component][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
      }

    } else if ( a.subdetId() == 5 ) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&

      component=TOBDetId(BC[i].detid).layer();
  
      if (BC[i].BadModule){
	NBadComponent[2][0][0]++;
	NBadComponent[2][component][0]++;
      }
      if (BC[i].BadFibers){ 
	NBadComponent[2][0][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
	NBadComponent[2][component][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
      }
      if (BC[i].BadApvs){
	NBadComponent[2][0][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
	NBadComponent[2][component][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
      }

    } else if ( a.subdetId() == 6 ) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&

      component=TECDetId(BC[i].detid).side()==2?TECDetId(BC[i].detid).wheel()+1:TECDetId(BC[i].detid).wheel()+10;
      
      if (BC[i].BadModule){
	NBadComponent[3][0][0]++;
	NBadComponent[3][component][0]++;
      }
      if (BC[i].BadFibers){ 
	NBadComponent[3][0][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
	NBadComponent[3][component][1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
      }
      if (BC[i].BadApvs){
	NBadComponent[3][0][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
	NBadComponent[3][component][2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	  ( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
      }
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
      component=TIDDetId(detid).side()==2?TIDDetId(detid).wheel()+1:TIDDetId(detid).wheel()+4;
    } else if ( a.subdetId() == 5 ) {
      subdet=2;
      component=TOBDetId(detid).layer();
    } else if ( a.subdetId() == 6 ) {
      subdet=3;
      component=TECDetId(detid).side()==2?TECDetId(detid).wheel()+1:TECDetId(detid).wheel()+10;
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
  ss << "Global Info\n------------";
  ss << "\nBadComponent \tModules \tFibers \tApvs\tStrips\n--------------------------------";
  ss << "\nTracker:\t\t"<<NTkBadComponent[0]<<"\t"<<NTkBadComponent[1]<<"\t"<<NTkBadComponent[2]<<"\t"<<NTkBadComponent[3];
  ss<< "\n";
  ss << "\nTIB:\t\t\t"<<NBadComponent[0][0][0]<<"\t"<<NBadComponent[0][0][1]<<"\t"<<NBadComponent[0][0][2]<<"\t"<<NBadComponent[0][0][3];
  ss << "\nTID:\t\t\t"<<NBadComponent[1][0][0]<<"\t"<<NBadComponent[1][0][1]<<"\t"<<NBadComponent[1][0][2]<<"\t"<<NBadComponent[1][0][3];
  ss << "\nTOB:\t\t\t"<<NBadComponent[2][0][0]<<"\t"<<NBadComponent[2][0][1]<<"\t"<<NBadComponent[2][0][2]<<"\t"<<NBadComponent[2][0][3];
  ss << "\nTEC:\t\t\t"<<NBadComponent[3][0][0]<<"\t"<<NBadComponent[3][0][1]<<"\t"<<NBadComponent[3][0][2]<<"\t"<<NBadComponent[3][0][3];
  ss << "\n";

  for (int i=1;i<5;++i)
    ss << "\nTIB Layer " << i << " :\t\t"<<NBadComponent[0][i][0]<<"\t"<<NBadComponent[0][i][1]<<"\t"<<NBadComponent[0][i][2]<<"\t"<<NBadComponent[0][i][3];
  ss << "\n";
  for (int i=1;i<4;++i)
    ss << "\nTID+ Disk " << i << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
  for (int i=4;i<7;++i)
    ss << "\nTID- Disk " << i-3 << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
  ss << "\n";
  for (int i=1;i<7;++i)
    ss << "\nTOB Layer " << i << " :\t\t"<<NBadComponent[2][i][0]<<"\t"<<NBadComponent[2][i][1]<<"\t"<<NBadComponent[2][i][2]<<"\t"<<NBadComponent[2][i][3];
  ss << "\n";
  for (int i=1;i<10;++i)
    ss << "\nTEC+ Disk " << i << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
  for (int i=10;i<19;++i)
    ss << "\nTEC- Disk " << i-9 << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];


  edm::LogInfo("SiStripQualityStatistics") << ss.str() << std::endl;
  
}
