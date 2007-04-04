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
// $Id: SiPixelHistogramId.cc,v 1.5 2007/03/21 16:39:28 chiochia Exp $
//

#include<boost/cstdint.hpp>
#include<iostream>
#include<sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"

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
std::string SiPixelHistogramId::createHistoId( std::string variable, uint32_t& rawId )
{
  std::string histoId;
  std::ostringstream rawIdString;
  rawIdString<<rawId;
  histoId = variable + separator_ + dataCollection_ + separator_  + rawIdString.str();

  return histoId;
}

