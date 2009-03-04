// -*- C++ -*-
//
// Package:     SiStripCommon
// Class  :     SiStripHistoId
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  dkcira
//         Created:  Wed Feb 22 16:07:58 CET 2006
// $Id: SiStripHistoId.cc,v 1.9 2009/02/18 14:28:02 maborgia Exp $
//

#include<iostream>
#include<sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

using namespace std;
using namespace edm;

SiStripHistoId::SiStripHistoId()
{
 separator1="__";
 separator2="__";
}


SiStripHistoId::~SiStripHistoId()
{
}


std::string SiStripHistoId::createHistoId(std::string description, std::string id_type,uint32_t component_id){
  unsigned int pos1 = description.find( separator1, 0 ); // check if std::string 'description' contains by mistake the 'separator1'
  unsigned int pos2 = description.find( separator2, 0 ); // check if std::string 'description' contains by mistake the 'separator2'
  if ( pos1 == std::string::npos && pos2 == std::string::npos ){ // ok, not found either separator
    if(id_type=="fed" || id_type=="det" || id_type=="fec"){ // ok! is one of the accepted id_type-s
      std::ostringstream compid; compid<<component_id; // use std::ostringstream for casting integer to std::string
      std::string local_histo_id = description + separator1 + id_type + separator2 + compid.str();
      return local_histo_id;
    }else{
      LogWarning("SiStripTkDQM|WrongInput")<<"no such type of component accepted: "<<id_type
                            <<" . id_type can be: fed, det, or fec."
                            <<"   Throwing exception";
      throw std::string("Exception thrown");
    }
  }else{
    LogWarning("SiStripTkDQM|WrongInput")<<"histogram description cannot contain: "<<separator1<<" or: "<<separator2
                          <<" histogram description = "<<description
                          <<" . Throwing exception";
    throw std::string("Exception thrown");
  }
}

std::string SiStripHistoId::createHistoLayer(std::string description, std::string id_type,std::string path,std::string flag){
  size_t pos1 = description.find( separator1, 0 ); // check if std::string 'description' contains by mistake the 'separator1'
  size_t pos2 = description.find( separator2, 0 ); // check if std::string 'description' contains by mistake the 'separator2'
  if ( pos1 == std::string::npos && pos2 == std::string::npos ){ // ok, not found either separator
    if(id_type=="fed" || id_type=="det" || id_type=="fec"  || id_type=="layer"){ // ok! is one of the accepted id_type-s
      std::ostringstream compid; compid<<path; // use std::ostringstream for casting integer to std::string
      std::string local_histo_id;
      if(flag.size() > 0)
	local_histo_id = description + separator1 + flag + separator2 + path;
      else 
	local_histo_id = description + separator2 + path;

      LogTrace("SiStripHistoId") << "Local_histo_ID " << local_histo_id << std::endl;
      return local_histo_id;
    }else{
      LogWarning("SiStripTkDQM|WrongInput")<<"no such type of component accepted: "<<id_type
                            <<" . id_type can be: fed, det, fec or layer."
                            <<"   Throwing exception";
      throw std::string("Exception thrown");
    }
  }else{
    LogWarning("SiStripTkDQM|WrongInput")<<"histogram description cannot contain: "<<separator1<<" or: "<<separator2
                          <<" histogram description = "<<description
                          <<" . Throwing exception";
    throw std::string("Exception thrown");
  }
}

std::string SiStripHistoId::getSubdetid(uint32_t id,bool flag_ring){
  char rest[1024];

  StripSubdetector subdet(id);
  if( subdet.subdetId() == StripSubdetector::TIB){
  // ---------------------------  TIB  --------------------------- //
    TIBDetId tib1 = TIBDetId(id);
    sprintf(rest,"TIB__layer__%d",tib1.layer());
  }else if( subdet.subdetId() == StripSubdetector::TID){
  // ---------------------------  TID  --------------------------- //
    TIDDetId tid1 = TIDDetId(id);
    sprintf(rest,"TID__side__%d__wheel__%d",tid1.side(),tid1.wheel());
  }else if(subdet.subdetId() == StripSubdetector::TOB){ 
  // ---------------------------  TOB  --------------------------- //
    TOBDetId tob1 = TOBDetId(id);
    sprintf(rest,"TOB__layer__%d",tob1.layer());
  }else if(subdet.subdetId() == StripSubdetector::TEC){
  // ---------------------------  TEC  --------------------------- //
    TECDetId tec1 = TECDetId(id);
    sprintf(rest,"TEC__side__%d__wheel__%d",tec1.side(),tec1.wheel());
  }else{
  // ---------------------------  ???  --------------------------- //
    edm::LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdet.subdetId()<<" no folder set!"<<std::endl;
    return 0;
  }
  
  if(flag_ring){
    if(subdet.subdetId() == StripSubdetector::TID){
      // ---------------------------  TID  --------------------------- //
      TIDDetId tid1 = TIDDetId(id);
      sprintf(rest,"TID__side__%d__ring__%d",tid1.side(),tid1.ring());
    }else if( subdet.subdetId() == StripSubdetector::TEC){
      // ---------------------------  TEC  --------------------------- //
      TECDetId tec1 = TECDetId(id);
      sprintf(rest,"TEC__side__%d__ring__%d",tec1.side(),tec1.ring());
    }
  }

  std::string rest1(rest);
  return rest1;
}

uint32_t SiStripHistoId::getComponentId(std::string histoid){
  uint32_t local_component_id;
  std::istringstream input(returnIdPart(histoid,3)); input >> local_component_id; // use std::istringstream for casting from std::string to uint32_t
  return local_component_id;
}


std::string SiStripHistoId::getComponentType(std::string histoid){
 return returnIdPart(histoid,2);
}


std::string SiStripHistoId::returnIdPart(std::string histoid, uint32_t whichpart){
  size_t length1=histoid.find(separator1,0);
  if(length1==std::string::npos){ // no separator1 found
    LogWarning("SiStripTkDQM|UnregularInput")<<"no regular histoid. Returning 0";
    return "0";
  }
  std::string part1 = histoid.substr(0,length1); // part of 'histoid' up to 'separator1'
  if(whichpart==1) return part1;
  std::string remain1 = histoid.substr(length1+separator1.size()); // rest of 'histoid' starting at end of 'separator1'
  size_t length2=remain1.find(separator2,0);
  if(length2==std::string::npos){ // no separator2 found
    LogWarning("SiStripTkDQM|UnregularInput")<<"no regular histoid. Returning 0";
    return "0";
  }
  std::string part2 = remain1.substr(0,length2); // part of 'remain1' up to 'separator2'
  if(whichpart==2) return part2;
  std::string part3 = remain1.substr(length2+separator2.size()); // rest of remain1 starting at end of 'separator2'
  if(whichpart==3) return part3;
  LogWarning("SiStripTkDQM|UnregularInput")<<"no such whichpart="<<whichpart<<" returning 0";
  return "0";
}

