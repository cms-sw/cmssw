// -*- C++ -*-
//
// Package:     SiStripCommon
// Class  :     SiStripFolderOrganizer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  dkcira
//         Created:  Thu Jan 26 23:52:43 CET 2006
// $Id: SiStripFolderOrganizer.cc,v 1.1 2006/02/09 19:02:26 gbruno Exp $
//

#include <iostream>
#include <sstream>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"

using namespace std;

SiStripFolderOrganizer::SiStripFolderOrganizer()
{
  TopFolderName="SiStrip";
  MechanicalFolderName="MechanicalView";
  ReadoutFolderName="ReadoutView";
  ControlFolderName="ControlView";
  sep ="/";
  // get a pointer to DaqMonitorBEInterface
  dbe_  = edm::Service<DaqMonitorBEInterface>().operator->();
}

SiStripFolderOrganizer::~SiStripFolderOrganizer()
{
}

void SiStripFolderOrganizer::setDetectorFolder(uint32_t rawdetid){
  string lokal_folder = TopFolderName + sep + MechanicalFolderName;
  if(rawdetid == 0 ){ // just MechanicalFolder if rawdetid==0;
    dbe_->setCurrentFolder(lokal_folder);
    return;
  }

  ostringstream rest;
  int subdetid = ((rawdetid>>25)&0x7);
  if(       subdetid==3 ){
    TIBDetId tib1 = TIBDetId(rawdetid);
    rest<<sep<<"TIB"<<sep<<"layer_"<<tib1.layer()<<sep<<"string_"<<(tib1.string()).at(2)<<sep<<"module_in_string_"<<tib1.module()<<sep <<"detector_"<<rawdetid;
  }else if( subdetid==4){
    TIDDetId tid1 = TIDDetId(rawdetid);
    rest<<sep<<"TID"<<sep<<"side_"<<tid1.side()<<sep<<"wheel_"<<tid1.wheel()<<sep<<"ring_"<<tid1.ring()<<sep<<"module_in_ring_"<<(tid1.module()).at(1)<<sep<<"detector_"<<rawdetid;
  }else if( subdetid==5){
    TOBDetId tob1 = TOBDetId(rawdetid);
    rest<<sep<<"TOB"<<sep<<"layer_"<<tob1.layer()<<sep<<"rod_"<<(tob1.rod()).at(1)<<sep<<"module_in_rod_"<<tob1.module()<<sep<<"detector_"<<rawdetid;
  }else if( subdetid==6){
    TECDetId tec1 = TECDetId(rawdetid);
    rest<<sep<<"TEC"<<sep<<"side_"<<tec1.side()<<sep<<"wheel_"<<tec1.wheel()<<sep<<"petal_"<<(tec1.petal()).at(1)<<sep<<"ring_"<<tec1.ring()<<sep<<"module_in_ring_"<<tec1.module()<<sep<<"detector_"<<rawdetid;
  }else{
    cout<<"SiStripFolderOrganizer::setDetectorFolder - no such subdetector type :"<<subdetid<<endl;
    cout<<"                           no folder set!"<<endl;
    return;
  }

  lokal_folder += rest.str();
  dbe_->setCurrentFolder(lokal_folder);
}


// string SiStripFolderOrganizer::getSiStripFolder(){
//   return TopFolderName;
// }
// 
// string SiStripFolderOrganizer::getReadoutFolder(unsigned int fed_id){
//   string lokal_folder;
//   if (fed_id==99999) {
//     lokal_folder = TopFolderName + sep + ReadoutFolderName;
//   }else{
//     ostringstream strfed;
//     strfed<<"FED_"<<fed_id;
//     lokal_folder = TopFolderName + sep + ReadoutFolderName + sep + strfed.str() ;
//   }
//   return lokal_folder;
// }
// 
// string SiStripFolderOrganizer::getTOBFolder(){
//   string lokal_folder = TopFolderName + sep + TOBFolderName;
//   return lokal_folder;
// }
// 
// string SiStripFolderOrganizer::getTIBFolder(){
//   string lokal_folder = TopFolderName + sep + TIBFolderName;
//   return lokal_folder;
// }
