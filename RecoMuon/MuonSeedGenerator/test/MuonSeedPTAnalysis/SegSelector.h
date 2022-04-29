#ifndef RecoMuon_SegSelector_H
#define RecoMuon_SegSelector_H

/** \class SegSelector
 *
 *  Author: S.C. Kao  - UC Riverside
 */

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/Common/interface/Handle.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4D.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment2D.h>
#include <DataFormats/DTRecHit/interface/DTRecHitCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecHit1D.h>

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include <vector>
#include <map>
#include <string>
#include <utility>

//class PSimHit;
class CSCLayer;
class CSCDetId;
class DTLayerId;
class DTSuperLayerId;
class DTChamberId;

// Sim_CSCSegments products
struct SimSegment {
  int chamber_type;  // DT =2 , CSC = 1
  CSCDetId csc_DetId;
  DTChamberId dt_DetId;
  LocalPoint sLocalOrg;
  GlobalVector sGlobalVec;
  GlobalPoint sGlobalOrg;
  std::vector<PSimHit> simhit_v;
};

class SegSelector {
public:
  /// Constructor
  explicit SegSelector(const edm::ParameterSet& pset);
  //SegSelector();

  /// Destructor
  virtual ~SegSelector();
  //~SegSelector();

  // Operations

  /// for EDAnalyzer
  //void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  // Functions
  /// return the sim segments sCSC_v and sDT_v
  std::vector<SimSegment> Sim_CSCSegments(int trkId,
                                          const edm::Handle<edm::PSimHitContainer> simHits,
                                          edm::ESHandle<CSCGeometry> cscGeom);
  std::vector<SimSegment> Sim_DTSegments(int trkId,
                                         const edm::Handle<edm::PSimHitContainer> simHits,
                                         edm::ESHandle<DTGeometry> dtGeom);
  /// return the reco segments cscseg_V and dtseg_V which map to the sim segment
  std::vector<CSCSegment> Select_CSCSeg(edm::Handle<CSCSegmentCollection> cscSeg,
                                        edm::ESHandle<CSCGeometry> cscGeom,
                                        std::vector<SimSegment> simseg);
  std::vector<DTRecSegment4D> Select_DTSeg(edm::Handle<DTRecSegment4DCollection> dtSeg,
                                           edm::ESHandle<DTGeometry> dtGeom,
                                           std::vector<SimSegment> simseg);

protected:
private:
  // Utility functions
  void CSCSimHitFit(edm::ESHandle<CSCGeometry> cscGeom);
  void DTSimHitFit(edm::ESHandle<DTGeometry> dtGeom);

  void LongCSCSegment(std::vector<CSCSegment> cscsegs);
  void LongDTSegment(std::vector<DTRecSegment4D> dtsegs);

  // sim segment output
  std::vector<SimSegment> sDT_v;
  std::vector<SimSegment> sCSC_v;

  // LongSegment
  std::vector<CSCSegment> longsegV;
  std::vector<DTRecSegment4D> longsegV1;

  // Select_DTSeg ouput
  std::vector<DTRecSegment4D> dtseg_V;
  // Select_CSCSeg output
  std::vector<CSCSegment> cscseg_V;

  // SimHitFit DT
  LocalVector LSimVec1;
  LocalPoint LSimOrg1;
  GlobalVector GSimVec1;
  GlobalPoint GSimOrg1;

  // SimHitFit variables
  std::vector<PSimHit> hit_V1;
  std::vector<PSimHit> hit_V;
  double par1[2];
  double par2[2];
  LocalVector LSimVec;
  LocalPoint LSimOrg;
  GlobalVector GSimVec;
  GlobalPoint GSimOrg;

  // Switch for debug output
  bool debug;

  std::string cscSegmentLabel;
  std::string recHitLabel;
  std::string dtSegmentLabel;
  std::string dtrecHitLabel;
  std::string simHitLabel;
  std::string simTrackLabel;
};

#endif
