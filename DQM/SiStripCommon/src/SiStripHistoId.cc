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
// $Id: SiStripHistoId.cc,v 1.18 2013/01/03 18:59:35 wmtan Exp $
//

#include<iostream>
#include<sstream>
#include<cstdio>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

SiStripHistoId::SiStripHistoId()
{
}


SiStripHistoId::~SiStripHistoId()
{
}


std::string SiStripHistoId::createHistoId(std::string description, std::string id_type,uint32_t component_id){
  size_t pos1 = description.find("__", 0 ); // check if std::string 'description' contains by mistake the 'separator1'
  size_t pos2 = description.find("__", 0 ); // check if std::string 'description' contains by mistake the 'separator2'
  std::string local_histo_id;
  std::ostringstream compid;
  compid<<component_id; // use std::ostringstream for casting integer to std::string
      
  if ( pos1 == std::string::npos && pos2 == std::string::npos ){ // ok, not found either separator
    if(id_type=="fed" || id_type=="det" || id_type=="fec"){ // ok! is one of the accepted id_type-s
      local_histo_id = description + "__" + id_type + "__" + compid.str();
    }else{
      local_histo_id = description + "__dummy__" + compid.str();
      edm::LogError("SiStripHistoId") <<" SiStripHistoId::WrongInput " 
				 <<" no such type of component accepted: "<<id_type
				 <<" id_type can be: fed, det, or fec.";      
    }
  }else{
    local_histo_id = description + "_dummy___" + id_type + "__" + compid.str();
    edm::LogError("SiStripHistoId") <<" SiStripHistoId::WrongInput " 
			       <<" histogram description cannot contain: __ or: __" 
			       <<" histogram description = "<<description;
  }
  return local_histo_id;
}

std::string SiStripHistoId::createHistoLayer(std::string description, std::string id_type,std::string path,std::string flag){
  size_t pos1 = description.find( "__", 0 ); // check if std::string 'description' contains by mistake the 'separator1'
  size_t pos2 = description.find( "__", 0 ); // check if std::string 'description' contains by mistake the 'separator2'
  std::string local_histo_id;
  if ( pos1 == std::string::npos && pos2 == std::string::npos ){ // ok, not found either separator
    if(id_type=="fed" || id_type=="det" || id_type=="fec"  || id_type=="layer"){ // ok! is one of the accepted id_type-s
      if(flag.size() > 0)
	local_histo_id = description + "__" + flag + "__" + path;
      else 
	local_histo_id = description + "__" + path;
      LogTrace("SiStripHistoId") << "Local_histo_ID " << local_histo_id << std::endl;
    }else{
      local_histo_id = description + "___dummy___" + path;
      edm::LogError("SiStripHistoId") <<" SiStripHistoId::WrongInput " 
				 <<" no such type of component accepted: "<<id_type
				 <<" id_type can be: fed, det, fec or layer ";
    }
  }else{
    local_histo_id = description + "_dummy___" + path;
    edm::LogWarning("SiStripHistoId") <<" SiStripHistoId::WrongInput " 
                                 <<" histogram description cannot contain: __ or: __"
				 <<" histogram description = "<<description;
  }
  return local_histo_id;
}

std::string SiStripHistoId::getSubdetid(uint32_t id, const TrackerTopology* tTopo, bool flag_ring){
  std::string rest1;
  
  const int buf_len = 50;
  char temp_str[buf_len];

  StripSubdetector subdet(id);
  if( subdet.subdetId() == StripSubdetector::TIB){
    // ---------------------------  TIB  --------------------------- //
    
    snprintf(temp_str, buf_len, "TIB__layer__%i", tTopo->tibLayer(id));
  }else if( subdet.subdetId() == StripSubdetector::TID){
    // ---------------------------  TID  --------------------------- //
    
    if (flag_ring) snprintf(temp_str, buf_len, "TID__side__%i__ring__%i", tTopo->tidSide(id), tTopo->tidRing(id));
    else snprintf(temp_str, buf_len, "TID__side__%i__wheel__%i", tTopo->tidSide(id), tTopo->tidWheel(id));
  }else if(subdet.subdetId() == StripSubdetector::TOB){ 
    // ---------------------------  TOB  --------------------------- //
    
    snprintf(temp_str, buf_len, "TOB__layer__%i",tTopo->tobLayer(id)); 
  }else if(subdet.subdetId() == StripSubdetector::TEC){
    // ---------------------------  TEC  --------------------------- //
    
    if (flag_ring) snprintf(temp_str, buf_len, "TEC__side__%i__ring__%i", tTopo->tecSide(id), tTopo->tecRing(id));
    else snprintf(temp_str, buf_len, "TEC__side__%i__wheel__%i", tTopo->tecSide(id), tTopo->tecWheel(id)); 
  }else{
    // ---------------------------  ???  --------------------------- //
    edm::LogError("SiStripTkDQM|WrongInput")<<"no such subdetector type :"<<subdet.subdetId()<<" no folder set!"<<std::endl;
    snprintf(temp_str,0,"%s","");  
  }
  
  return std::string(temp_str);
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
  size_t length1=histoid.find("__",0);
  if(length1==std::string::npos){ // no separator1 found
    edm::LogWarning("SiStripTkDQM|UnregularInput")<<"no regular histoid. Returning 0";
    return "0";
  }
  std::string part1 = histoid.substr(0,length1); // part of 'histoid' up to 'separator1'
  if(whichpart==1) return part1;
  std::string remain1 = histoid.substr(length1+2); // rest of 'histoid' starting at end of 'separator1'
  size_t length2=remain1.find("__",0);
  if(length2==std::string::npos){ // no separator2 found
    edm::LogWarning("SiStripTkDQM|UnregularInput")<<"no regular histoid. Returning 0";
    return "0";
  }
  std::string part2 = remain1.substr(0,length2); // part of 'remain1' up to 'separator2'
  if(whichpart==2) return part2;
  std::string part3 = remain1.substr(length2+2); // rest of remain1 starting at end of 'separator2'
  if(whichpart==3) return part3;
  edm::LogWarning("SiStripTkDQM|UnregularInput")<<"no such whichpart="<<whichpart<<" returning 0";
  return "0";
}

