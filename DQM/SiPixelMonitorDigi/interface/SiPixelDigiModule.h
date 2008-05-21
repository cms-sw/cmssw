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
// $Id: SiPixelDigiModule.h,v 1.7 2008/04/16 17:08:01 merkelp Exp $
//
//
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
  void book(const edm::ParameterSet& iConfig);
  /// Fill histograms
  void fill(const edm::DetSetVector<PixelDigi> & input);
  
 private:

  uint32_t id_;
  int ncols_;
  int nrows_;
  MonitorElement* meNDigis_;
  MonitorElement* meADC_;
  MonitorElement* mePixDigis_;
  
};
#endif
