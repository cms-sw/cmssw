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
// $Id: SiStripFolderOrganizer.cc,v 1.6 2006/04/23 13:26:20 dkcira Exp $
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


string SiStripFolderOrganizer::getSiStripFolder(){
   return TopFolderName;
}


void SiStripFolderOrganizer::setSiStripFolder(){
   dbe_->setCurrentFolder(TopFolderName);
   return;
}


string SiStripFolderOrganizer::getSiStripTopControlFolder(){
   string lokal_folder = TopFolderName + ControlFolderName;
   return lokal_folder;
}


void SiStripFolderOrganizer::setSiStripTopControlFolder(){
   string lokal_folder = TopFolderName + ControlFolderName;
   dbe_->setCurrentFolder(lokal_folder);
   return;
}


string SiStripFolderOrganizer::getSiStripControlFolder(
                                   // unsigned short crate,
                                   unsigned short slot,
                                   unsigned short ring,
                                   unsigned short addr,
                                   unsigned short chan
                                   // unsigned short i2c
                                   ) {
  stringstream lokal_folder;
  lokal_folder << getSiStripTopControlFolder();
  //   if ( crate != all_ ) {// if ==all_ then remain in top control folder
  //     lokal_folder << sep << "FecCrate" << crate;
  if ( slot != all_ ) {
    lokal_folder << sep << "FecSlot" << slot;
    if ( ring != all_ ) {
      lokal_folder << sep << "FecRing" << ring;
      if ( addr != all_ ) {
	lokal_folder << sep << "CcuAddr" << addr;
	if ( chan != all_ ) {
	  lokal_folder << sep << "CcuChan" << chan;
	  // 	    if ( i2c != all_ ) {
	  // 	      lokal_folder << sep << "I2cAddr" << i2c;
	  // 	    }
	}
      }
    }
  }
  //   }
  return lokal_folder.str();
}


void SiStripFolderOrganizer::setSiStripControlFolder(
                                   // unsigned short crate,
                                   unsigned short slot,
                                   unsigned short ring,
                                   unsigned short addr,
                                   unsigned short chan
                                   // unsigned short i2c
                                   ) {
 string lokal_folder = getSiStripControlFolder(slot, ring, addr, chan);
 dbe_->setCurrentFolder(lokal_folder);
 return;
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
    LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdetid<<" no folder set!"<<endl;
    return;
  }

  lokal_folder += rest.str();
  dbe_->setCurrentFolder(lokal_folder);
}

