// -*- C++ -*-
//
// Package:     SiPixelCommon
// Class  :     SiPixelHistogramId
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  chiochia
//         Created:  Wed Feb 22 16:07:58 CET 2006
// $Id: SiPixelHistogramId.cc,v 1.4 2010/11/29 20:41:58 wmtan Exp $
//

#include<iostream>
#include<sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"

using namespace edm;

/// Constructor
SiPixelHistogramId::SiPixelHistogramId() : 
  dataCollection_("defaultData"),
  separator_("_")
{
}
/// Constructor with collection
SiPixelHistogramId::SiPixelHistogramId(std::string dataCollection) : 
  dataCollection_(dataCollection),
  separator_("_")
{
}

/// Destructor
SiPixelHistogramId::~SiPixelHistogramId()
{
}
/// Create Histogram Id
std::string SiPixelHistogramId::setHistoId( std::string variable, uint32_t& rawId )
{
  std::string histoId;
  std::ostringstream rawIdString;
  rawIdString<<rawId;
  histoId = variable + separator_ + dataCollection_ + separator_  + rawIdString.str();

  return histoId;
}
/// get Data Collection
std::string SiPixelHistogramId::getDataCollection( std::string histoid ) {
  return returnIdPart(histoid,2);
}
/// get Raw Id
uint32_t SiPixelHistogramId::getRawId( std::string histoid ) {
  uint32_t local_component_id;
  std::istringstream input(returnIdPart(histoid,3)); input >> local_component_id; 
  return local_component_id;
}
/// get Part
std::string SiPixelHistogramId::returnIdPart(std::string histoid, uint32_t whichpart){

  size_t length1=histoid.find(separator_,0);
  if(length1==std::string::npos){ // no separator1 found
    LogWarning("PixelDQM")<<"SiPixelHistogramId::returnIdPart - no regular histoid. Returning 0";
    return "0";
  }
  std::string part1 = histoid.substr(0,length1); // part of 'histoid' up to 'separator1'
  if(whichpart==1) return part1;
  std::string remain1 = histoid.substr(length1+separator_.size()); // rest of 'histoid' starting at end of 'separator1'
  size_t length2=remain1.find(separator_,0);
  if(length2==std::string::npos){ // no separator2 found
    LogWarning("PixelDQM")<<"SiPixelHistogramId::returnIdPart - no regular histoid. Returning 0";
    return "0";
  }
  std::string part2 = remain1.substr(0,length2); // part of 'remain1' up to 'separator2'
  if(whichpart==2) return part2;
  std::string part3 = remain1.substr(length2+separator_.size()); // rest of remain1 starting at end of 'separator2'
  if(whichpart==3) return part3;
  LogWarning("PixelDQM")<<"SiPixelHistogramId::returnIdPart - no such whichpart="<<whichpart<<" returning 0";
  return "0";
}
