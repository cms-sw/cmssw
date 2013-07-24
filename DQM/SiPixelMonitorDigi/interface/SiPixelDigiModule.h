#ifndef SiPixelMonitorDigi_SiPixelDigiModule_h
#define SiPixelMonitorDigi_SiPixelDigiModule_h
// -*- C++ -*-
//
// Package:    SiPixelMonitorDigi
// Class:      SiPixelDigiModule
// 
/**\class 

 Description: Digi monitoring elements for a Pixel sensor

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelDigiModule.h,v 1.21 2013/02/04 13:18:43 merkelp Exp $
//
//
//  Updated by: Lukas Wehrli
//  for pixel offline DQM 

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include <boost/cstdint.hpp>

class SiPixelDigiModule {        

 public:

  /// Default constructor
  SiPixelDigiModule();
  /// Constructor with raw DetId
  SiPixelDigiModule(const uint32_t& id);
  /// Constructor with raw DetId and sensor size
  SiPixelDigiModule(const uint32_t& id, const int& ncols, const int& nrows);
  /// Destructor
  ~SiPixelDigiModule();

  typedef edm::DetSet<PixelDigi>::const_iterator    DigiIterator;

  /// Book histograms
  void book(const edm::ParameterSet& iConfig, int type=0, bool twoD=true, bool hiRes=false, bool reducedSet=false, bool additInfo=false, bool isUpgrade=false);
  /// Fill histograms
//  int fill(const edm::DetSetVector<PixelDigi> & input, bool modon=true, 
//						 bool ladon=false, bool layon=false, bool phion=false, 
//						 bool bladeon=false, bool diskon=false, bool ringon=false, 
//						 bool twoD=true, bool reducedSet=false, bool twoDimModOn = true, bool twoDimOnlyLayDisk = false,
//						 int &nDigisA, int &nDigisB);
  int fill(const edm::DetSetVector<PixelDigi> & input, const bool modon, 
						 const bool ladon, const bool layon, const bool phion, 
						 const bool bladeon, const bool diskon, const bool ringon, 
						 const bool twoD, const bool reducedSet, const bool twoDimModOn, const bool twoDimOnlyLayDisk,
						 int &nDigisA, int &nDigisB, bool isUpgrade);
  
 private:

  uint32_t id_;
  int ncols_;
  int nrows_;
  MonitorElement* meNDigis_;
  MonitorElement* meADC_;
  MonitorElement* mePixDigis_;
  MonitorElement* mePixDigis_px_;
  MonitorElement* mePixDigis_py_;

  //barrel:
  MonitorElement* meNDigisLad_;
  MonitorElement* meADCLad_;
  MonitorElement* mePixDigisLad_;
  MonitorElement* mePixDigisLad_px_;
  MonitorElement* mePixDigisLad_py_;

  MonitorElement* meNDigisLay_;
  MonitorElement* meADCLay_;
  MonitorElement* mePixDigisLay_;
  MonitorElement* mePixRocsLay_;
  MonitorElement* meZeroOccRocsLay_;
  MonitorElement* mePixDigisLay_px_;
  MonitorElement* mePixDigisLay_py_;

  MonitorElement* meNDigisPhi_;
  MonitorElement* meADCPhi_;
  MonitorElement* mePixDigisPhi_;
  MonitorElement* mePixDigisPhi_px_;
  MonitorElement* mePixDigisPhi_py_;

  //forward:
  MonitorElement* meNDigisBlade_;
  MonitorElement* meADCBlade_;

  MonitorElement* meNDigisDisk_;
  MonitorElement* meADCDisk_;
  MonitorElement* mePixDigisDisk_;
  MonitorElement* mePixRocsDisk_;
  MonitorElement* meZeroOccRocsDisk_;

  MonitorElement* meNDigisRing_;
  MonitorElement* meADCRing_;
  MonitorElement* mePixDigisRing_;
  MonitorElement* mePixDigisRing_px_;
  MonitorElement* mePixDigisRing_py_;
  
  //int nEventDigis_;

};
#endif
