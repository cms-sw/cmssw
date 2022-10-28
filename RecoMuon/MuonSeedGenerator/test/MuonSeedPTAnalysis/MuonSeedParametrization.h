#ifndef RecoMuon_MuonSeedParametrization_H
#define RecoMuon_MuonSeedParametrization_H

/** \class SeedParametrization
 *
 *  Author: S.C. Kao  - UC Riverside
 */

#include "SegSelector.h"
#include "MuonSeedParameterHisto.h"
#include "MuonSeedParaFillHisto.h"
#include "MuonSeeddPhiScale.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
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
#include <DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h>
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
#include <DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>

#include <vector>
#include <map>
#include <string>
#include <utility>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

//class PSimHit;
class TFile;
class CSCLayer;
class CSCDetId;
class DTLayerId;
class DTSuperLayerId;
class DTChamberId;
class SegSelector;
class MuonSeedParaFillHisto;
class MuonSeeddPhiScale;

class MuonSeedParametrization : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  MuonSeedParametrization(const edm::ParameterSet &pset);

  /// Destructor
  virtual ~MuonSeedParametrization();

  // Operations
  /// Perform the real analysis
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup);

protected:
private:
  SegSelector *recsegSelector;
  MuonSeedParaFillHisto *HistoFill;
  MuonSeeddPhiScale *ScaledPhi;

  // Utility functions
  void CSCsegment_stat(edm::Handle<CSCSegmentCollection> cscSeg);
  void DTsegment_stat(edm::Handle<DTRecSegment4DCollection> dtSeg);

  void CSCRecHit_stat(edm::Handle<CSCRecHit2DCollection> cscrechit, edm::ESHandle<CSCGeometry> cscGeom);
  void DTRecHit_stat(edm::Handle<DTRecHitCollection> dtrechit, edm::ESHandle<DTGeometry> dtGeom);

  bool SameChamber(CSCDetId SimDetId, CSCDetId SegDetId);

  void SimInfo(const edm::Handle<edm::SimTrackContainer> simTracks,
               const edm::Handle<edm::PSimHitContainer> dsimHits,
               const edm::Handle<edm::PSimHitContainer> csimHits,
               edm::ESHandle<DTGeometry> dtGeom,
               edm::ESHandle<CSCGeometry> cscGeom);

  void FromCSCSeg(std::vector<CSCSegment> cscSeg, edm::ESHandle<CSCGeometry> cscGeom, std::vector<SimSegment> seg);
  void FromCSCSingleSeg(std::vector<CSCSegment> cscSeg,
                        edm::ESHandle<CSCGeometry> cscGeom,
                        std::vector<SimSegment> seg);
  void FromDTSeg(std::vector<DTRecSegment4D> dtSeg, edm::ESHandle<DTGeometry> dtGeom, std::vector<SimSegment> seg);
  void FromDTSingleSeg(std::vector<DTRecSegment4D> dtSeg,
                       edm::ESHandle<DTGeometry> dtGeom,
                       std::vector<SimSegment> seg);
  void FromOverlap();

  // Histograms
  H2DRecHit1 *h_all;
  H2DRecHit2 *h_csc;
  H2DRecHit3 *h_dt;
  H2DRecHit4 *hME1[15];
  H2DRecHit5 *hMB1[26];
  H2DRecHit6 *hME2[8];
  H2DRecHit7 *hMB2[12];
  H2DRecHit10 *hOL1[6];

  // The file which will store the histos
  TFile *theFile;

  //cscsegment_stat output
  int cscseg_stat[6];
  int cscseg_stat1[6];
  //dtsegment_stat output
  int dtseg_stat[6];
  int dtseg_stat1[6];
  int dt2Dseg_stat[6];

  // SeedfromRecHit
  //std::vector<CSCRecHit2D> csc_rh;
  int cscrh_sum[6];
  int dtrh_sum[6];

  // Sim info
  double pt1[5];
  double pa[5];
  double eta_c;
  double eta_d;
  double eta_trk;
  double theQ;
  double etaLc[5];
  double etaLd[5];
  double ptLossC[5];
  double ptLossD[5];

  // dPhi and dEta for CSC
  double PhiV1[2][5];
  double EtaV1[2][5];
  double PhiP1[2][5];
  double EtaP1[2][5];
  double dPhiV1[2][5][5];
  double dEtaV1[2][5][5];
  double dPhiP1[2][5][5];
  double dEtaP1[2][5][5];
  double chi2_dof1[5];
  /// dphi and eta for CSC single segment
  bool MEPath[2][5][4];
  double ME_phi[2][5][4];
  double ME_eta[2][5][4];

  // dPhi and dEta for DT
  double PhiV3[2][5];
  double EtaV3[2][5];
  double dPhiV3[2][5][5];
  double dEtaV3[2][5][5];
  double PhiP3[2][5];
  double EtaP3[2][5];
  double dPhiP3[2][5][5];
  double dEtaP3[2][5][5];
  double chi2_dof3[5];
  /// dphi and eta for DT single segment
  bool MBPath[2][5][3];
  double MB_phi[2][5][3];
  double MB_eta[2][5][3];

  // dphi & Eta for Overlap region
  double dPhiV2[2][5][5];
  double dEtaV2[2][5][5];
  double dPhiP2[2][5][5];
  double dEtaP2[2][5][5];

  // Switch for debug output
  bool debug;
  bool scale;

  std::string rootFileName;
  std::string cscSegmentLabel;
  std::string recHitLabel;
  std::string dtSegmentLabel;
  std::string dt2DSegmentLabel;
  std::string dtrecHitLabel;
  std::string simHitLabel;
  std::string simTrackLabel;
  std::string muonseedLabel;

  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken;
};

#endif
