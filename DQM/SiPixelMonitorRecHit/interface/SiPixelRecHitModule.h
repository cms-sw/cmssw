#ifndef SiPixelMonitorRecHits_SiPixelRecHitModule_h
#define SiPixelMonitorRecHits_SiPixelRecHitModule_h
// -*- C++ -*-
//
// Package:    SiPixelMonitorRecHits
// Class:      SiPixelRecHitModule
// 
/**\class 

 Description: RecHits monitoring elements for a single Pixel
detector segment (detID)

 Implementation:
//	This package was marginally adapted from 
//	SiPixelDigiModule
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelDigiModule.h,v 1.6 2007/04/04 13:57:04 chiochia Exp $
//
//  Adapted by: Keith Rose
//  for use in SiPixelMonitorRecHit package

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/cstdint.hpp>

class SiPixelRecHitModule {        

 public:

  /// Default constructor
  SiPixelRecHitModule();
  /// Constructor with raw DetId
  SiPixelRecHitModule(const uint32_t& id);
  /// Destructor
  ~SiPixelRecHitModule();

  // typedef edm::DetSet<PixelRecHit>::const_iterator  RecHitsIterator;

  /// Book histograms
  void book(const edm::ParameterSet& iConfig);
  /// Fill histograms
  void fill(const float& rechit_x, const float& rechit_y, const float& x_res, const float& y_res, const float& x_pull, const float& y_pull);
  
 private:

  uint32_t id_;
  MonitorElement* meXYPos_;
  MonitorElement* meXRes_;
  MonitorElement* meYRes_;
  MonitorElement* meXPull_;
  MonitorElement* meYPull_;
  
};
#endif
