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
// $Id: SiStripFolderOrganizer.cc,v 1.3 2006/03/29 21:41:38 dkcira Exp $
//

#include <iostream>
#include <sstream>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"

using namespace std;
using namespace edm;

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
  if(rawdetid == 0 ){ // just top MechanicalFolder if rawdetid==0;
    dbe_->setCurrentFolder(lokal_folder);
    return;
  }

  ostringstream rest;
  int subdetid = ((rawdetid>>25)&0x7);
  if(       subdetid==3 ){
  // ---------------------------  TIB  --------------------------- //
    TIBDetId tib1 = TIBDetId(rawdetid);
    vector<string> str_subnames(3); // translate numering of tib string-structure in (character) strings
    if((tib1.string()).at(0)==0){str_subnames[0]="backward_strings";}else{str_subnames[0]="forward_strings";}
    if((tib1.string()).at(1)==0){str_subnames[1]="internal_strings";}else{str_subnames[1]="external_strings";}
    rest<<sep<<"TIB"<<sep<<"layer_"<<tib1.layer()<<sep<<str_subnames[0]<<sep<<str_subnames[1]<<sep<<"string_"<<(tib1.string()).at(2)<<sep <<"module_"<<rawdetid;
  }else if( subdetid==4){
  // ---------------------------  TID  --------------------------- //
    TIDDetId tid1 = TIDDetId(rawdetid);
    string stereo_mono;
    if(tid1.stereo()==0){stereo_mono="mono_modules";}else{stereo_mono="stereo_modules";}
    rest<<sep<<"TID"<<sep<<"side_"<<tid1.side()<<sep<<"wheel_"<<tid1.wheel()<<sep<<"ring_"<<tid1.ring()<<sep<<stereo_mono<<sep<<"module_"<<rawdetid;
  }else if( subdetid==5){
  // ---------------------------  TOB  --------------------------- //
    TOBDetId tob1 = TOBDetId(rawdetid);
    string bkw_frw;
    if((tob1.rod()).at(0)==0){bkw_frw="backward_rods";}else{bkw_frw="forward_rods";}
    rest<<sep<<"TOB"<<sep<<"layer_"<<tob1.layer()<<sep<<bkw_frw<<sep<<"rod_"<<(tob1.rod()).at(1)<<sep<<"module_"<<rawdetid;
  }else if( subdetid==6){
  // ---------------------------  TEC  --------------------------- //
    TECDetId tec1 = TECDetId(rawdetid);
    string petal_bkw_frw;     if((tec1.petal()).at(0)==0){petal_bkw_frw="backward_petals";}{petal_bkw_frw="forward_petals";}
    string fec_stereo_mono; if(tec1.stereo()==0){fec_stereo_mono="mono_modules";}{fec_stereo_mono="stereo_modules";}
    rest<<sep<<"TEC"<<sep<<"side_"<<tec1.side()<<sep<<"wheel_"<<tec1.wheel()<<sep<<petal_bkw_frw<<sep<<"petal_"<<(tec1.petal()).at(1)<<sep<<"ring_"<<tec1.ring()<<sep<<fec_stereo_mono<<sep<<"module_"<<rawdetid;
  }else{
  // ---------------------------  ???  --------------------------- //
    LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdetid<<" no folder set!"<<endl;
    return;
  }

  lokal_folder += rest.str();
  dbe_->setCurrentFolder(lokal_folder);
}


// old methods
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
