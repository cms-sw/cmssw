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
//
//
//  Updated by: Lukas Wehrli
//  for pixel offline DQM

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include <cstdint>

class SiPixelDigiModule {
public:
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;

  /// Default constructor
  SiPixelDigiModule();
  /// Constructor with raw DetId
  SiPixelDigiModule(const uint32_t& id);
  /// Constructor with raw DetId and sensor size
  SiPixelDigiModule(const uint32_t& id, const int& ncols, const int& nrows);
  /// Destructor
  ~SiPixelDigiModule();

  typedef edm::DetSet<PixelDigi>::const_iterator DigiIterator;

  /// Book histograms
  void book(const edm::ParameterSet& iConfig,
            const edm::EventSetup& iSetup,
            DQMStore::IBooker& iBooker,
            int type = 0,
            bool twoD = true,
            bool hiRes = false,
            bool reducedSet = false,
            bool additInfo = false,
            bool isUpgrade = false);
  /// Fill histograms
  //  int fill(const edm::DetSetVector<PixelDigi> & input, bool modon=true,
  //           bool ladon=false, bool layon=false, bool phion=false,
  //           bool bladeon=false, bool diskon=false, bool ringon=false,
  //           bool twoD=true, bool reducedSet=false, bool twoDimModOn = true, bool twoDimOnlyLayDisk = false,
  //           int &nDigisA, int &nDigisB);
  int fill(const edm::DetSetVector<PixelDigi>& input,
           const edm::EventSetup& iSetup,
           MonitorElement* combBarrel,
           MonitorElement* chanBarrel,
           std::vector<MonitorElement*>& chanBarrelL,
           MonitorElement* combEndcap,
           const bool modon,
           const bool ladon,
           const bool layon,
           const bool phion,
           const bool bladeon,
           const bool diskon,
           const bool ringon,
           const bool twoD,
           const bool reducedSet,
           const bool twoDimModOn,
           const bool twoDimOnlyLayDisk,
           int& nDigisA,
           int& nDigisB,
           bool isUpgrade);
  void
  resetRocMap();  // This is to move the rocmap reset from the Source to the Module where the map is booked. Necessary for multithread safety.
  std::pair<int, int> getZeroLoEffROCs();  // Moved from Souce.cc. Gets number of zero and low eff ROCs from each module.
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
  MonitorElement* mePixRocsLay_ = nullptr;
  MonitorElement* meZeroOccRocsLay_ = nullptr;
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
  MonitorElement* mePixRocsDisk_ = nullptr;
  MonitorElement* meZeroOccRocsDisk_ = nullptr;

  MonitorElement* meNDigisRing_;
  MonitorElement* meADCRing_;
  MonitorElement* mePixDigisRing_;
  MonitorElement* mePixDigisRing_px_;
  MonitorElement* mePixDigisRing_py_;

  //int nEventDigis_;
};
#endif
