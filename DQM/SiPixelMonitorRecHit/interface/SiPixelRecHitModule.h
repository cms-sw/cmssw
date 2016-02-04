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
// $Id: SiPixelRecHitModule.h,v 1.8 2009/06/19 09:30:00 merkelp Exp $
//
//  Adapted by: Keith Rose
//  for use in SiPixelMonitorRecHit package
//  Updated by: Lukas Wehrli
//  for pixel offline DQM 

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
  void book(const edm::ParameterSet& iConfig, int type=0, bool twoD=true, 
            bool reducedSet=false);
  /// Fill histograms
  void fill(const float& rechit_x, const float& rechit_y, const int& sizeX, 
            const int& sizeY, const float& lerr_x, const float& lerr_y, 
	    bool modon=true, bool ladon=false, bool layon=false, 
	    bool phion=false, bool bladeon=false, bool diskon=false, 
	    bool ringon=false, bool twoD=true, bool reducedSet=false);
  void nfill(const int& nrec, bool modon=true, bool ladon=false, 
             bool layon=false, bool phion=false, bool bladeon=false, 
	     bool diskon=false, bool ringon=false);  

 private:

  uint32_t id_;
  MonitorElement* meXYPos_;
  MonitorElement* meXYPos_px_;
  MonitorElement* meXYPos_py_;
  MonitorElement* meClustX_;
  MonitorElement* meClustY_;  
  MonitorElement* meErrorX_;
  MonitorElement* meErrorY_;  
  MonitorElement* menRecHits_;
  //barrel
  MonitorElement* meXYPosLad_;
  MonitorElement* meXYPosLad_px_;
  MonitorElement* meXYPosLad_py_;
  MonitorElement* meClustXLad_;
  MonitorElement* meClustYLad_;
  MonitorElement* meErrorXLad_;
  MonitorElement* meErrorYLad_; 
  MonitorElement* menRecHitsLad_;

  MonitorElement* meXYPosLay_;
  MonitorElement* meXYPosLay_px_;
  MonitorElement* meXYPosLay_py_;
  MonitorElement* meClustXLay_;
  MonitorElement* meClustYLay_;
  MonitorElement* meErrorXLay_;
  MonitorElement* meErrorYLay_; 
  MonitorElement* menRecHitsLay_;

  MonitorElement* meXYPosPhi_;
  MonitorElement* meXYPosPhi_px_;
  MonitorElement* meXYPosPhi_py_;
  MonitorElement* meClustXPhi_;
  MonitorElement* meClustYPhi_;
  MonitorElement* meErrorXPhi_;
  MonitorElement* meErrorYPhi_; 
  MonitorElement* menRecHitsPhi_;

  //forward
  MonitorElement* meClustXBlade_;
  MonitorElement* meClustYBlade_;  
  MonitorElement* meErrorXBlade_;
  MonitorElement* meErrorYBlade_; 
  MonitorElement* menRecHitsBlade_;

  MonitorElement* meClustXDisk_;
  MonitorElement* meClustYDisk_;  
  MonitorElement* meErrorXDisk_;
  MonitorElement* meErrorYDisk_; 
  MonitorElement* menRecHitsDisk_;

  MonitorElement* meXYPosRing_;
  MonitorElement* meXYPosRing_px_;
  MonitorElement* meXYPosRing_py_;
  MonitorElement* meClustXRing_;
  MonitorElement* meClustYRing_;
  MonitorElement* meErrorXRing_;
  MonitorElement* meErrorYRing_; 
  MonitorElement* menRecHitsRing_;
};
#endif
