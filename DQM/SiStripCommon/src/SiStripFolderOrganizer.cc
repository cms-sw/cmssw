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

// $Id: SiStripFolderOrganizer.cc,v 1.26 2009/09/18 17:26:41 dutta Exp $
//

#include <iostream>
#include <sstream>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"

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
  // get a pointer to DQMStore
  dbe_  = edm::Service<DQMStore>().operator->();
}


SiStripFolderOrganizer::~SiStripFolderOrganizer()
{
}

void SiStripFolderOrganizer::setSiStripFolderName(string name){
  TopFolderName = name;
}

std::string SiStripFolderOrganizer::getSiStripFolder(){
   return TopFolderName;
}


void SiStripFolderOrganizer::setSiStripFolder(){
   dbe_->setCurrentFolder(TopFolderName);
   return;
}


std::string SiStripFolderOrganizer::getSiStripTopControlFolder(){
   std::string lokal_folder = TopFolderName + ControlFolderName;
   return lokal_folder;
}


void SiStripFolderOrganizer::setSiStripTopControlFolder(){
   std::string lokal_folder = TopFolderName + ControlFolderName;
   dbe_->setCurrentFolder(lokal_folder);
   return;
}


std::string SiStripFolderOrganizer::getSiStripControlFolder(
                                   // unsigned short crate,
                                   unsigned short slot,
                                   unsigned short ring,
                                   unsigned short addr,
                                   unsigned short chan
                                   // unsigned short i2c
                                   ) {
  std::stringstream lokal_folder;
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
  string folder_name = lokal_folder.str(); 
  return folder_name;
}


void SiStripFolderOrganizer::setSiStripControlFolder(
                                   // unsigned short crate,
                                   unsigned short slot,
                                   unsigned short ring,
                                   unsigned short addr,
                                   unsigned short chan
                                   // unsigned short i2c
                                   ) {
 std::string lokal_folder = getSiStripControlFolder(slot, ring, addr, chan);
 dbe_->setCurrentFolder(lokal_folder);
 return;
}

std::pair<std::string,int32_t> SiStripFolderOrganizer::GetSubDetAndLayer(const uint32_t& detid, bool ring_flag){
  std::string cSubDet;
  int32_t layer=0;
  switch(StripSubdetector::SubDetector(StripSubdetector(detid).subdetId()))
    {
    case StripSubdetector::TIB:
      cSubDet="TIB";
      layer=TIBDetId(detid).layer();
      break;
    case StripSubdetector::TOB:
      cSubDet="TOB";
      layer=TOBDetId(detid).layer();
      break;
    case StripSubdetector::TID:
      cSubDet="TID";
      if(ring_flag)
	layer=TIDDetId(detid).ring() * ( TIDDetId(detid).side()==1 ? -1 : +1);
      else
	layer=TIDDetId(detid).wheel() * ( TIDDetId(detid).side()==1 ? -1 : +1);
      break;
    case StripSubdetector::TEC:
      cSubDet="TEC";
      if(ring_flag)
	layer=TECDetId(detid).ring() * ( TECDetId(detid).side()==1 ? -1 : +1);
      else
	layer=TECDetId(detid).wheel() * ( TECDetId(detid).side()==1 ? -1 : +1);
      break;
    default:
      edm::LogWarning("SiStripMonitorTrack") << "WARNING!!! this detid does not belong to tracker" << std::endl;
    }
  return std::make_pair(cSubDet,layer);
}


void SiStripFolderOrganizer::setDetectorFolder(uint32_t rawdetid){
  string folder_name;
  getFolderName(rawdetid, folder_name);
  dbe_->setCurrentFolder(folder_name);
}

void SiStripFolderOrganizer::getSubDetLayerFolderName(std::stringstream& ss, SiStripDetId::SubDetector subDet, uint32_t layer, uint32_t side){
  ss << TopFolderName << sep << MechanicalFolderName;

  if(subDet == SiStripDetId::TIB){
    ss << sep << "TIB" << sep << "layer_" << layer << sep;
  } else if(subDet == SiStripDetId::TID){
    ss << sep << "TID" << sep << "side_" << side << sep << "wheel_" << layer << sep;
  } else if( subDet == SiStripDetId::TOB){
    ss << sep << "TOB" << sep << "layer_" << layer << sep;
  }else if(subDet == SiStripDetId::TEC){
    ss << sep << "TEC" << sep << "side_" << side << sep << "wheel_" << layer << sep;
  }else{
    // ---------------------------  ???  --------------------------- //
    LogWarning("SiStripTkDQM|WrongInput")<<"no such SubDet :"<< subDet <<" no folder set!"<<endl;
  }
}


void SiStripFolderOrganizer::getFolderName(int32_t rawdetid, string& lokal_folder){
  lokal_folder = ""; 
  if(rawdetid == 0 ){ // just top MechanicalFolder if rawdetid==0;
    return;
  }
  std::stringstream rest;
  SiStripDetId stripdet = SiStripDetId(rawdetid);
  
  if (stripdet.subDetector() == SiStripDetId::TIB){
  // ---------------------------  TIB  --------------------------- //
    TIBDetId tib = TIBDetId(rawdetid);
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tib.layerNumber());
    if (tib.isZMinusSide())      rest << "backward_strings" << sep;
    else                         rest << "forward_strings"  << sep;
    if (tib.isExternalString())  rest << "external_strings" << sep;
    else                         rest << "internal_strings" << sep;
    rest << "string_" << tib.stringNumber() << sep << "module_" << rawdetid;
  } else if(stripdet.subDetector() == SiStripDetId::TID){
  // ---------------------------  TID  --------------------------- //
    TIDDetId tid = TIDDetId(rawdetid);
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tid.wheel(),tid.side());
    rest<< "ring_"  << tid.ring() << sep;

    if (tid.isStereo()) rest << "stereo_modules" << sep;
    else                rest << "mono_modules" << sep;
    rest  << "module_" << rawdetid;
  } else if( stripdet.subDetector() == SiStripDetId::TOB){
  // ---------------------------  TOB  --------------------------- //
    TOBDetId tob = TOBDetId(rawdetid);
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tob.layerNumber());
    if (tob.isZMinusSide()) rest << "backward_rods" << sep;
    else                    rest << "forward_rods" << sep;
    rest << "rod_" << tob.rodNumber() << sep<< "module_" << rawdetid;
  }else if(stripdet.subDetector() == SiStripDetId::TEC){
  // ---------------------------  TEC  --------------------------- //
    TECDetId tec = TECDetId(rawdetid);
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tec.wheel(),tec.side());
    if (tec.isBackPetal()) rest << "backward_petals" << sep;
    else                   rest << "forward_petals" << sep;

    rest << "petal_" << tec.petalNumber() << sep
         << "ring_"<< tec.ringNumber() << sep;

    if (tec.isStereo())    rest << "stereo_modules" << sep;
    else                   rest << "mono_modules" << sep;

    rest << "module_" << rawdetid;
  }else{
     // ---------------------------  ???  --------------------------- //
    LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<stripdet.subDetector() <<" no folder set!"<<endl;
    return;
  }
  lokal_folder += rest.str();

}

void SiStripFolderOrganizer::setLayerFolder(uint32_t rawdetid, int32_t layer, bool ring_flag){
  std::string lokal_folder = TopFolderName + sep + MechanicalFolderName;
  if(rawdetid == 0 ){ // just top MechanicalFolder if rawdetid==0;
    dbe_->setCurrentFolder(lokal_folder);
    return;
  }

  std::ostringstream rest;
  SiStripDetId stripdet = SiStripDetId(rawdetid);
  if(stripdet.subDetector() == SiStripDetId::TIB ){
  // ---------------------------  TIB  --------------------------- //
    TIBDetId tib1 = TIBDetId(rawdetid);
    int tib_layer = tib1.layer();
    if (abs(layer)  != tib_layer) {
      LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tib1.layer() <<endl;
      return;
    }
    rest<<sep<<"TIB"<<sep<<"layer_"<<tib1.layer();
  }else if(stripdet.subDetector() == SiStripDetId::TID){
  // ---------------------------  TID  --------------------------- //
    TIDDetId tid1 = TIDDetId(rawdetid);
    int tid_ring = tid1.ring();
    if(ring_flag){
      if(abs(layer) != tid_ring) {
	LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tid1.ring() <<endl;
	return;
      }
      rest<<sep<<"TID"<<sep<<"side_"<<tid1.side()<<sep<<"ring_"<<tid1.ring();
    }else{
      int tid_wheel = tid1.wheel();
      if (abs(layer)  != tid_wheel) {
	LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tid1.wheel() <<endl;
	return;
      }
      rest<<sep<<"TID"<<sep<<"side_"<<tid1.side()<<sep<<"wheel_"<<tid1.wheel();
    }
  }else if(stripdet.subDetector() == SiStripDetId::TOB){
  // ---------------------------  TOB  --------------------------- //
    TOBDetId tob1 = TOBDetId(rawdetid);
    int tob_layer = tob1.layer();
    if (abs(layer)  != tob_layer) {
      LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tob1.layer() <<endl;
      return;
    }
    rest<<sep<<"TOB"<<sep<<"layer_"<<tob1.layer();
  }else if( stripdet.subDetector() == SiStripDetId::TEC){
  // ---------------------------  TEC  --------------------------- //
    TECDetId tec1 = TECDetId(rawdetid);
    if(ring_flag){
      int tec_ring = tec1.ring(); 
      if (abs(layer)  != tec_ring) {
	LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tec1.ring() <<endl;
	return;
      }
      rest<<sep<<"TEC"<<sep<<"side_"<<tec1.side()<<sep<<"ring_"<<tec1.ring();
    }else{
      int tec_wheel = tec1.wheel();
      if (abs(layer)  != tec_wheel) {
	LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tec1.wheel() <<endl;
	return;
      }
      rest<<sep<<"TEC"<<sep<<"side_"<<tec1.side()<<sep<<"wheel_"<<tec1.wheel();
    }
  }else{
  // ---------------------------  ???  --------------------------- //
    LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<stripdet.subDetector()<<" no folder set!"<<endl;
    return;
  }

  lokal_folder += rest.str();
  dbe_->setCurrentFolder(lokal_folder);
}

void SiStripFolderOrganizer::getSubDetFolder(const uint32_t& detid, std::string& folder_name){

  std::string subdet_folder;
  switch(StripSubdetector::SubDetector(StripSubdetector(detid).subdetId()))
    {
    case StripSubdetector::TIB:
      subdet_folder="TIB";
      break;
    case StripSubdetector::TOB:
      subdet_folder="TOB";
      break;
    case StripSubdetector::TID:
      if (TIDDetId(detid).side() == 2) {
	subdet_folder = "TID/side_2";
      } else if (TIDDetId(detid).side() == 1) {
	subdet_folder = "TID/side_1";
      }
      break;
    case StripSubdetector::TEC:
      if (TECDetId(detid).side() == 2) {
	subdet_folder = "TEC/side_2";
      } else if (TECDetId(detid).side() == 1) {
	subdet_folder = "TEC/side_1";
      }
      break;
    default:
      {
	edm::LogWarning("SiStripCommon") << "WARNING!!! this detid does not belong to tracker" << std::endl;
        subdet_folder = "";
      } 
    }
  folder_name = TopFolderName + sep + MechanicalFolderName + sep + subdet_folder;
}
//
// -- Get the name of Subdetector Layer folder
//
void SiStripFolderOrganizer::getLayerFolderName(std::stringstream& ss, uint32_t rawdetid,bool ring_flag){
  ss << TopFolderName + sep + MechanicalFolderName;
  if(rawdetid == 0 ){ // just top MechanicalFolder if rawdetid==0;
    return;
  }

  SiStripDetId stripdet = SiStripDetId(rawdetid);
  if(stripdet.subDetector() == SiStripDetId::TIB ){
  // ---------------------------  TIB  --------------------------- //
    TIBDetId tib1 = TIBDetId(rawdetid);
    ss<<sep<<"TIB"<<sep<<"layer_"<<tib1.layer();
  }else if(stripdet.subDetector() == SiStripDetId::TID){
  // ---------------------------  TID  --------------------------- //
    TIDDetId tid1 = TIDDetId(rawdetid);
    if(ring_flag){
      ss<<sep<<"TID"<<sep<<"side_"<<tid1.side()<<sep<<"ring_"<<tid1.ring();
    }else{
      ss<<sep<<"TID"<<sep<<"side_"<<tid1.side()<<sep<<"wheel_"<<tid1.wheel();
    }
  }else if(stripdet.subDetector() == SiStripDetId::TOB){
  // ---------------------------  TOB  --------------------------- //
    TOBDetId tob1 = TOBDetId(rawdetid);
    ss<<sep<<"TOB"<<sep<<"layer_"<<tob1.layer();
  }else if( stripdet.subDetector() == SiStripDetId::TEC){
  // ---------------------------  TEC  --------------------------- //
    TECDetId tec1 = TECDetId(rawdetid);
    if(ring_flag){
      ss<<sep<<"TEC"<<sep<<"side_"<<tec1.side()<<sep<<"ring_"<<tec1.ring();
    }else{
      ss<<sep<<"TEC"<<sep<<"side_"<<tec1.side()<<sep<<"wheel_"<<tec1.wheel();
    }
  }else{
  // ---------------------------  ???  --------------------------- //
    LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<stripdet.subDetector()<<" no folder set!"<<endl;
    return;
  }
}
