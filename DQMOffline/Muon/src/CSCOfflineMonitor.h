#ifndef RecoLocalMuon_CSCOfflineMonitor_H
#define RecoLocalMuon_CSCOfflineMonitor_H

/** \class CSCOfflineMonitor
 *
 * Simple package for offline CSC DQM based on RecoLocalMuon/CSCValidation:
 *    DIGIS
 *    recHits
 *    segments
 *
 * This program merely unpacks collections and fills
 * a few simple histograms.  The idea is to compare
 * the histograms for one offline release and another
 * and look for unexpected differences.
 *
 * Michael Schmitt, Northwestern University, July 2007
 */


// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class CSCOfflineMonitor : public edm::EDAnalyzer {
public:
  /// Constructor
  CSCOfflineMonitor(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~CSCOfflineMonitor();

  // Operations
  void beginJob(edm::EventSetup const& iSetup);
  void endJob(void); 
 
  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);


protected:

private: 

  edm::ParameterSet param;

  // some useful functions
  float      fitX(HepMatrix sp, HepMatrix ep);
  float      getTiming(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip);
  float      getSignal(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip);
  int        typeIndex(CSCDetId id);

  // DQM
  DQMStore* dbe;

  // Wire digis
  MonitorElement *hWireAll;
  MonitorElement *hWireTBinAll;
  MonitorElement *hWirenGroupsTotal;
  MonitorElement *hWireCodeBroad;
  std::vector<MonitorElement*> hWireLayer;
  std::vector<MonitorElement*> hWireWire;
  std::vector<MonitorElement*> hWireCodeNarrow;

  // Strip Digis
  MonitorElement *hStripAll;
  MonitorElement *hStripNFired;
  MonitorElement *hStripCodeBroad;
  std::vector<MonitorElement*> hStripCodeNarrow;
  std::vector<MonitorElement*> hStripLayer;
  std::vector<MonitorElement*> hStripStrip;

  // Pedestal Noise
  MonitorElement *hStripPedAll; 
  std::vector<MonitorElement*> hStripPed; 
  //MonitorElement *hPedvsStrip;

  // recHits
  MonitorElement *hRHCodeBroad;
  MonitorElement *hRHnrechits;
  std::vector<MonitorElement*> hRHCodeNarrow;
  std::vector<MonitorElement*> hRHLayer;
  std::vector<MonitorElement*> hRHX;
  std::vector<MonitorElement*> hRHY;
  std::vector<MonitorElement*> hRHGlobal;
  std::vector<MonitorElement*> hRHResid;
  std::vector<MonitorElement*> hSResid;
  std::vector<MonitorElement*> hRHSumQ;
  std::vector<MonitorElement*> hRHRatioQ;
  std::vector<MonitorElement*> hRHTiming;

  // Segments
  MonitorElement *hSCodeBroad;
  std::vector<MonitorElement*> hSCodeNarrow;
  std::vector<MonitorElement*> hSnHits;
  std::vector<MonitorElement*> hSTheta;
  std::vector<MonitorElement*> hSGlobal;
  MonitorElement *hSnhitsAll;
  MonitorElement *hSChiSqProb;
  MonitorElement *hSGlobalTheta;
  MonitorElement *hSGlobalPhi;
  MonitorElement *hSnSegments;

  // occupancy histos
  MonitorElement *hOWires;
  MonitorElement *hOStrips;
  MonitorElement *hORecHits;
  MonitorElement *hOSegments;



};
#endif
