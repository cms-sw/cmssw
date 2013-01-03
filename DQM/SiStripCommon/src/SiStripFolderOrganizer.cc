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

// $Id: SiStripFolderOrganizer.cc,v 1.31 2013/01/02 17:37:22 wmtan Exp $
//

#include <iostream>
#include <sstream>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#define CONTROL_FOLDER_NAME "ControlView"
#define MECHANICAL_FOLDER_NAME "MechanicalView"
#define SEP "/"

SiStripFolderOrganizer::SiStripFolderOrganizer()
{
  TopFolderName="SiStrip";
  // get a pointer to DQMStore
  dbe_  = edm::Service<DQMStore>().operator->();
}


SiStripFolderOrganizer::~SiStripFolderOrganizer()
{
}

void SiStripFolderOrganizer::setSiStripFolderName(std::string name){
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
   std::string lokal_folder = TopFolderName + CONTROL_FOLDER_NAME;
   return lokal_folder;
}


void SiStripFolderOrganizer::setSiStripTopControlFolder(){
   std::string lokal_folder = TopFolderName + CONTROL_FOLDER_NAME;
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
  //     lokal_folder << SEP << "FecCrate" << crate;
  if ( slot != all_ ) {
    lokal_folder << SEP << "FecSlot" << slot;
    if ( ring != all_ ) {
      lokal_folder << SEP << "FecRing" << ring;
      if ( addr != all_ ) {
	lokal_folder << SEP << "CcuAddr" << addr;
	if ( chan != all_ ) {
	  lokal_folder << SEP << "CcuChan" << chan;
	  // 	    if ( i2c != all_ ) {
	  // 	      lokal_folder << SEP << "I2cAddr" << i2c;
	  // 	    }
	}
      }
    }
  }
  //   }
  std::string folder_name = lokal_folder.str(); 
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

std::pair<std::string,int32_t> SiStripFolderOrganizer::GetSubDetAndLayer(const uint32_t& detid, const TrackerTopology* tTopo, bool ring_flag){
  std::string cSubDet;
  int32_t layer=0;
  switch(StripSubdetector::SubDetector(StripSubdetector(detid).subdetId()))
    {
    case StripSubdetector::TIB:
      cSubDet="TIB";
      layer=tTopo->tibLayer(detid);
      break;
    case StripSubdetector::TOB:
      cSubDet="TOB";
      layer=tTopo->tobLayer(detid);
      break;
    case StripSubdetector::TID:
      cSubDet="TID";
      if(ring_flag)
	layer=tTopo->tidRing(detid) * ( tTopo->tidSide(detid)==1 ? -1 : +1);
      else
	layer=tTopo->tidWheel(detid) * ( tTopo->tidSide(detid)==1 ? -1 : +1);
      break;
    case StripSubdetector::TEC:
      cSubDet="TEC";
      if(ring_flag)
	layer=tTopo->tecRing(detid) * ( tTopo->tecSide(detid)==1 ? -1 : +1);
      else
	layer=tTopo->tecWheel(detid) * ( tTopo->tecSide(detid)==1 ? -1 : +1);
      break;
    default:
      edm::LogWarning("SiStripMonitorTrack") << "WARNING!!! this detid does not belong to tracker" << std::endl;
    }
  return std::make_pair(cSubDet,layer);
}


void SiStripFolderOrganizer::setDetectorFolder(uint32_t rawdetid, const TrackerTopology* tTopo){
  std::string folder_name;
  getFolderName(rawdetid, tTopo, folder_name);
  dbe_->setCurrentFolder(folder_name);
}

void SiStripFolderOrganizer::getSubDetLayerFolderName(std::stringstream& ss, SiStripDetId::SubDetector subDet, uint32_t layer, uint32_t side){
  ss << TopFolderName << SEP << MECHANICAL_FOLDER_NAME;

  if(subDet == SiStripDetId::TIB){
    ss << SEP << "TIB" << SEP << "layer_" << layer << SEP;
  } else if(subDet == SiStripDetId::TID){
    ss << SEP << "TID" << SEP << "side_" << side << SEP << "wheel_" << layer << SEP;
  } else if( subDet == SiStripDetId::TOB){
    ss << SEP << "TOB" << SEP << "layer_" << layer << SEP;
  }else if(subDet == SiStripDetId::TEC){
    ss << SEP << "TEC" << SEP << "side_" << side << SEP << "wheel_" << layer << SEP;
  }else{
    // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")<<"no such SubDet :"<< subDet <<" no folder set!"<<std::endl;
  }
}


void SiStripFolderOrganizer::getFolderName(int32_t rawdetid, const TrackerTopology* tTopo, std::string& lokal_folder){
  lokal_folder = ""; 
  if(rawdetid == 0 ){ // just top MechanicalFolder if rawdetid==0;
    return;
  }
  std::stringstream rest;
  SiStripDetId stripdet = SiStripDetId(rawdetid);
  
  if (stripdet.subDetector() == SiStripDetId::TIB){
  // ---------------------------  TIB  --------------------------- //
    
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tTopo->tibLayer(rawdetid));

    if (tTopo->tibIsZMinusSide(rawdetid))      rest << "backward_strings" << SEP;
    else                         rest << "forward_strings"  << SEP;
    if (tTopo->tibIsExternalString(rawdetid))  rest << "external_strings" << SEP;
    else                         rest << "internal_strings" << SEP;
    rest << "string_" << tTopo->tibString(rawdetid) << SEP << "module_" << rawdetid;
  } else if(stripdet.subDetector() == SiStripDetId::TID){
  // ---------------------------  TID  --------------------------- //
    
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tTopo->tidWheel(rawdetid),tTopo->tidSide(rawdetid));
    rest<< "ring_"  << tTopo->tidRing(rawdetid) << SEP;

    if (tTopo->tidIsStereo(rawdetid)) rest << "stereo_modules" << SEP;
    else                rest << "mono_modules" << SEP;
    rest  << "module_" << rawdetid;
  } else if( stripdet.subDetector() == SiStripDetId::TOB){
  // ---------------------------  TOB  --------------------------- //
    
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tTopo->tobLayer(rawdetid));
    if (tTopo->tobIsZMinusSide(rawdetid)) rest << "backward_rods" << SEP;
    else                    rest << "forward_rods" << SEP;
    rest << "rod_" << tTopo->tobRod(rawdetid) << SEP<< "module_" << rawdetid;
  }else if(stripdet.subDetector() == SiStripDetId::TEC){
  // ---------------------------  TEC  --------------------------- //
    
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tTopo->tecWheel(rawdetid),tTopo->tecSide(rawdetid));
    if (tTopo->tecIsBackPetal(rawdetid)) rest << "backward_petals" << SEP;
    else                   rest << "forward_petals" << SEP;

    rest << "petal_" << tTopo->tecPetalNumber(rawdetid) << SEP
         << "ring_"<< tTopo->tecRing(rawdetid) << SEP;

    if (tTopo->tecIsStereo(rawdetid))    rest << "stereo_modules" << SEP;
    else                   rest << "mono_modules" << SEP;

    rest << "module_" << rawdetid;
  }else{
     // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<stripdet.subDetector() <<" no folder set!"<<std::endl;
    return;
  }
  lokal_folder += rest.str();

}

void SiStripFolderOrganizer::setLayerFolder(uint32_t rawdetid, const TrackerTopology* tTopo, int32_t layer, bool ring_flag){
  std::string lokal_folder = TopFolderName + SEP + MECHANICAL_FOLDER_NAME;
  if(rawdetid == 0 ){ // just top MechanicalFolder if rawdetid==0;
    dbe_->setCurrentFolder(lokal_folder);
    return;
  }

  std::ostringstream rest;
  SiStripDetId stripdet = SiStripDetId(rawdetid);
  if(stripdet.subDetector() == SiStripDetId::TIB ){
  // ---------------------------  TIB  --------------------------- //
    
    int tib_layer = tTopo->tibLayer(rawdetid);
    if (abs(layer)  != tib_layer) {
      edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tTopo->tibLayer(rawdetid) <<std::endl;
      return;
    }
    rest<<SEP<<"TIB"<<SEP<<"layer_"<<tTopo->tibLayer(rawdetid);
  }else if(stripdet.subDetector() == SiStripDetId::TID){
  // ---------------------------  TID  --------------------------- //
    
    int tid_ring = tTopo->tidRing(rawdetid);
    if(ring_flag){
      if(abs(layer) != tid_ring) {
	edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tTopo->tidRing(rawdetid) <<std::endl;
	return;
      }
      rest<<SEP<<"TID"<<SEP<<"side_"<<tTopo->tidSide(rawdetid)<<SEP<<"ring_"<<tTopo->tidRing(rawdetid);
    }else{
      int tid_wheel = tTopo->tidWheel(rawdetid);
      if (abs(layer)  != tid_wheel) {
	edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tTopo->tidWheel(rawdetid) <<std::endl;
	return;
      }
      rest<<SEP<<"TID"<<SEP<<"side_"<<tTopo->tidSide(rawdetid)<<SEP<<"wheel_"<<tTopo->tidWheel(rawdetid);
    }
  }else if(stripdet.subDetector() == SiStripDetId::TOB){
  // ---------------------------  TOB  --------------------------- //
    
    int tob_layer = tTopo->tobLayer(rawdetid);
    if (abs(layer)  != tob_layer) {
      edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tTopo->tobLayer(rawdetid) <<std::endl;
      return;
    }
    rest<<SEP<<"TOB"<<SEP<<"layer_"<<tTopo->tobLayer(rawdetid);
  }else if( stripdet.subDetector() == SiStripDetId::TEC){
  // ---------------------------  TEC  --------------------------- //
    
    if(ring_flag){
      int tec_ring = tTopo->tecRing(rawdetid); 
      if (abs(layer)  != tec_ring) {
	edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tTopo->tecRing(rawdetid) <<std::endl;
	return;
      }
      rest<<SEP<<"TEC"<<SEP<<"side_"<<tTopo->tecSide(rawdetid)<<SEP<<"ring_"<<tTopo->tecRing(rawdetid);
    }else{
      int tec_wheel = tTopo->tecWheel(rawdetid);
      if (abs(layer)  != tec_wheel) {
	edm::LogWarning("SiStripTkDQM|Layer mismatch!!!")<< " expect "<<  abs(layer) << " but getting " << tTopo->tecWheel(rawdetid) <<std::endl;
	return;
      }
      rest<<SEP<<"TEC"<<SEP<<"side_"<<tTopo->tecSide(rawdetid)<<SEP<<"wheel_"<<tTopo->tecWheel(rawdetid);
    }
  }else{
  // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<stripdet.subDetector()<<" no folder set!"<<std::endl;
    return;
  }

  lokal_folder += rest.str();
  dbe_->setCurrentFolder(lokal_folder);
}

void SiStripFolderOrganizer::getSubDetFolder(const uint32_t& detid, const TrackerTopology* tTopo, std::string& folder_name){

  std::pair<std::string, std::string> subdet_and_tag = getSubDetFolderAndTag(detid, tTopo); 
  folder_name = subdet_and_tag.first;
}
//
// -- Get the name of Subdetector Layer folder
//
void SiStripFolderOrganizer::getLayerFolderName(std::stringstream& ss, uint32_t rawdetid, const TrackerTopology* tTopo, bool ring_flag){
  ss << TopFolderName + SEP + MECHANICAL_FOLDER_NAME;
  if(rawdetid == 0 ){ // just top MechanicalFolder if rawdetid==0;
    return;
  }

  SiStripDetId stripdet = SiStripDetId(rawdetid);
  if(stripdet.subDetector() == SiStripDetId::TIB ){
  // ---------------------------  TIB  --------------------------- //
    
    ss<<SEP<<"TIB"<<SEP<<"layer_"<<tTopo->tibLayer(rawdetid);
  }else if(stripdet.subDetector() == SiStripDetId::TID){
  // ---------------------------  TID  --------------------------- //
    
    if(ring_flag){
      ss<<SEP<<"TID"<<SEP<<"side_"<<tTopo->tidSide(rawdetid)<<SEP<<"ring_"<<tTopo->tidRing(rawdetid);
    }else{
      ss<<SEP<<"TID"<<SEP<<"side_"<<tTopo->tidSide(rawdetid)<<SEP<<"wheel_"<<tTopo->tidWheel(rawdetid);
    }
  }else if(stripdet.subDetector() == SiStripDetId::TOB){
  // ---------------------------  TOB  --------------------------- //
    
    ss<<SEP<<"TOB"<<SEP<<"layer_"<<tTopo->tobLayer(rawdetid);
  }else if( stripdet.subDetector() == SiStripDetId::TEC){
  // ---------------------------  TEC  --------------------------- //
    
    if(ring_flag){
      ss<<SEP<<"TEC"<<SEP<<"side_"<<tTopo->tecSide(rawdetid)<<SEP<<"ring_"<<tTopo->tecRing(rawdetid);
    }else{
      ss<<SEP<<"TEC"<<SEP<<"side_"<<tTopo->tecSide(rawdetid)<<SEP<<"wheel_"<<tTopo->tecWheel(rawdetid);
    }
  }else{
  // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<stripdet.subDetector()<<" no folder set!"<<std::endl;
    return;
  }
}
//
// -- Get Subdetector Folder name and the Tag
//
std::pair<std::string, std::string> SiStripFolderOrganizer::getSubDetFolderAndTag(const uint32_t& detid, const TrackerTopology* tTopo) {
  std::pair<std::string, std::string> result;
  result.first = TopFolderName + SEP MECHANICAL_FOLDER_NAME SEP;
  std::string subdet_folder;
  switch(StripSubdetector::SubDetector(StripSubdetector(detid).subdetId()))
    {
    case StripSubdetector::TIB:
      subdet_folder = "TIB";
      result.second = subdet_folder;
      break;
    case StripSubdetector::TOB:
      subdet_folder = "TOB";
      result.second = subdet_folder;
      break;
    case StripSubdetector::TID:
      if (tTopo->tidSide(detid) == 2) {
        subdet_folder = "TID/side_2";
	result.second = "TID__side__2";
      } else if (tTopo->tidSide(detid) == 1) {
        subdet_folder = "TID/side_1";
	result.second = "TID__side__1";
      }
      break;
    case StripSubdetector::TEC:
      if (tTopo->tecSide(detid) == 2) {
        subdet_folder = "TEC/side_2";
	result.second = "TEC__side__2";
      } else if (tTopo->tecSide(detid) == 1) {
        subdet_folder = "TEC/side_1";
	result.second = "TEC__side__1";
      }
      break;
    default:
      {
	edm::LogWarning("SiStripCommon") << "WARNING!!! this detid does not belong to tracker" << std::endl;
        subdet_folder = "";
      }
    }
  result.first += subdet_folder;
  return result; 
}


// This is the deprecated version, still needed for now.
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

void SiStripFolderOrganizer::getFolderName(int32_t rawdetid, std::string& lokal_folder){
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
    if (tib.isZMinusSide())      rest << "backward_strings" << SEP;
    else                         rest << "forward_strings"  << SEP;
    if (tib.isExternalString())  rest << "external_strings" << SEP;
    else                         rest << "internal_strings" << SEP;
    rest << "string_" << tib.stringNumber() << SEP << "module_" << rawdetid;
  } else if(stripdet.subDetector() == SiStripDetId::TID){
  // ---------------------------  TID  --------------------------- //
    TIDDetId tid = TIDDetId(rawdetid);
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tid.wheel(),tid.side());
    rest<< "ring_"  << tid.ring() << SEP;

    if (tid.isStereo()) rest << "stereo_modules" << SEP;
    else                rest << "mono_modules" << SEP;
    rest  << "module_" << rawdetid;
  } else if( stripdet.subDetector() == SiStripDetId::TOB){
  // ---------------------------  TOB  --------------------------- //
    TOBDetId tob = TOBDetId(rawdetid);
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tob.layerNumber());
    if (tob.isZMinusSide()) rest << "backward_rods" << SEP;
    else                    rest << "forward_rods" << SEP;
    rest << "rod_" << tob.rodNumber() << SEP<< "module_" << rawdetid;
  }else if(stripdet.subDetector() == SiStripDetId::TEC){
  // ---------------------------  TEC  --------------------------- //
    TECDetId tec = TECDetId(rawdetid);
    getSubDetLayerFolderName(rest,stripdet.subDetector(),tec.wheel(),tec.side());
    if (tec.isBackPetal()) rest << "backward_petals" << SEP;
    else                   rest << "forward_petals" << SEP;

    rest << "petal_" << tec.petalNumber() << SEP
         << "ring_"<< tec.ringNumber() << SEP;

    if (tec.isStereo())    rest << "stereo_modules" << SEP;
    else                   rest << "mono_modules" << SEP;

    rest << "module_" << rawdetid;
  }else{
     // ---------------------------  ???  --------------------------- //
    edm::LogWarning("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<stripdet.subDetector() <<" no folder set!"<<std::endl;
    return;
  }
  lokal_folder += rest.str();
}
