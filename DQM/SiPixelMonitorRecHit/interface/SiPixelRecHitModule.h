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
// $Id: SiPixelRecHitModule.h,v 1.1 2007/10/10 03:37:27 krose Exp $
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
  void fill(const float& rechit_x, const float& rechit_y, const int& sizeX, const int& sizeY);
  
 private:

  uint32_t id_;
  MonitorElement* meXYPos_;
  MonitorElement* meClustX_;
  MonitorElement* meClustY_;  
};
#endif
