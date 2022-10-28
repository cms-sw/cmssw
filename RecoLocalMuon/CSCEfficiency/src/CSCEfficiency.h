#ifndef RecoLocalMuon_CSCEfficiency_H
#define RecoLocalMuon_CSCEfficiency_H

/** \class CSCEfficiency
 *
 * Efficiency calculations
 * Stoyan Stoynev, Northwestern University
 */

// how many of the headers below are not needed?...
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "TFile.h"
#include "TVector3.h"
#include "TProfile.h"
#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"

#include <iostream>
#include <iomanip>
#include <fstream>
//#include "Math/Interpolator.h"

#include <vector>
#include <map>
#include <string>

#define SQR(x) ((x) * (x))

#define LastCh 36
#define FirstCh 1
#define NumCh (LastCh - FirstCh + 1)

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class TriggerNames;
}  // namespace edm

class TFile;
class CSCLayer;
class CSCDetId;

class CSCEfficiency : public edm::one::EDFilter<> {
public:
  /// Constructor
  CSCEfficiency(const edm::ParameterSet &pset);

  /// Destructor
  ~CSCEfficiency() override;

private:
  //---- analysis + filter
  bool filter(edm::Event &event, const edm::EventSetup &eventSetup) override;

  //---- (input) parameters
  //---- Root file name
  std::string rootFileName;

  //---- digi/object tokens

  edm::EDGetTokenT<CSCWireDigiCollection> wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection> al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection> cl_token;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> co_token;
  edm::EDGetTokenT<CSCRecHit2DCollection> rh_token;
  edm::EDGetTokenT<CSCSegmentCollection> se_token;

  edm::EDGetTokenT<edm::View<reco::Track> > tk_token;
  edm::EDGetTokenT<edm::PSimHitContainer> sh_token;

  edm::EDGetTokenT<edm::TriggerResults> ht_token;

  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geomToken_;

  //
  unsigned int printout_NEvents;
  bool isData;
  bool isIPdata;
  bool isBeamdata;
  bool getAbsoluteEfficiency;
  bool useDigis;
  double distanceFromDeadZone;
  double minP;
  double maxP;
  double maxNormChi2;
  unsigned int minTrackHits;

  bool applyIPangleCuts;
  double local_DY_DZ_Max;
  double local_DY_DZ_Min;
  double local_DX_DZ_Max;

  // trigger
  bool useTrigger;
  std::vector<std::string> myTriggers;
  std::vector<int> pointToTriggers;
  bool andOr;

  //---- The muon service
  MuonServiceProxy *theService;
  //---- The root file for the histograms
  TFile *theFile;
  //---- printalot debug output
  bool printalot;
  //---- counter
  int nEventsAnalyzed;
  //---- magnetic field
  bool magField;
  //---- track direction
  bool alongZ;
  //---- filter variable
  bool passTheEvent;

  //---- Variables
  //---- LCTs
  bool allCLCT[2][4][4][NumCh];     //endcap/station/ring/chamber
  bool allALCT[2][4][4][NumCh];     //endcap/station/ring/chamber
  bool allCorrLCT[2][4][4][NumCh];  //endcap/station/ring/chamber

  //----  Strips: strip number and ADCPeak
  std::vector<std::pair<int, float> > allStrips[2][4][4][NumCh][6];  //endcap/station/ring/chamber/layer

  //----Wires: WG number and Y-position, time bin
  std::vector<std::pair<std::pair<int, float>, int> > allWG[2][4][4][NumCh][6];  //endcap/station/ring/chamber/layer

  //SetOfSimHits  (*all_SimHits)[2][4][4][ NumCh];
  //---- Simhits
  std::vector<std::pair<LocalPoint, int> > allSimhits[2][4][4][NumCh][6];  //endcap/station/ring/chamber/layer

  //rechits
  //SetOfRecHits  (*all_RecHits)[2][4][4][ NumCh];
  std::vector<std::pair<LocalPoint, bool> > allRechits[2][4][4][NumCh][6];  //endcap/station/ring/chamber/layer

  // segments
  std::vector<std::pair<LocalPoint, LocalVector> > allSegments[2][4][4][NumCh];  //endcap/station/ring/chamber

  // empty chambers
  bool emptyChambers[2][4][4][NumCh];  //endcap/station/ring/chamber

  //---- Functions
  void fillDigiInfo(edm::Handle<CSCALCTDigiCollection> &alcts,
                    edm::Handle<CSCCLCTDigiCollection> &clcts,
                    edm::Handle<CSCCorrelatedLCTDigiCollection> &correlatedlcts,
                    edm::Handle<CSCWireDigiCollection> &wires,
                    edm::Handle<CSCStripDigiCollection> &strips,
                    edm::Handle<edm::PSimHitContainer> &simhits,
                    edm::Handle<CSCRecHit2DCollection> &rechits,
                    edm::Handle<CSCSegmentCollection> &segments,
                    edm::ESHandle<CSCGeometry> &cscGeom);
  void fillLCT_info(edm::Handle<CSCALCTDigiCollection> &alcts,
                    edm::Handle<CSCCLCTDigiCollection> &clcts,
                    edm::Handle<CSCCorrelatedLCTDigiCollection> &correlatedlcts);
  void fillWG_info(edm::Handle<CSCWireDigiCollection> &wires, edm::ESHandle<CSCGeometry> &cscGeom);
  void fillStrips_info(edm::Handle<CSCStripDigiCollection> &strips);
  void fillRechitsSegments_info(edm::Handle<CSCRecHit2DCollection> &rechits,
                                edm::Handle<CSCSegmentCollection> &segments,
                                edm::ESHandle<CSCGeometry> &cscGeom);
  void fillSimhit_info(edm::Handle<edm::PSimHitContainer> &simHits);

  void ringCandidates(int station, float absEta, std::map<std::string, bool> &chamberTypes);
  void chamberCandidates(int station, int ring, float phi, std::vector<int> &coupleOfChambers);

  bool efficienciesPerChamber(CSCDetId &id, const CSCChamber *cscChamber, FreeTrajectoryState &ftsChamber);
  bool stripWire_Efficiencies(CSCDetId &cscDetId, FreeTrajectoryState &ftsChamber);
  bool recHitSegment_Efficiencies(CSCDetId &cscDetId, const CSCChamber *cscChamber, FreeTrajectoryState &ftsChamber);
  bool recSimHitEfficiency(CSCDetId &id, FreeTrajectoryState &ftsChamber);
  //
  void returnTypes(CSCDetId &id, int &ec, int &st, int &rg, int &ch, int &secondRing);
  //
  void getFromFTS(const FreeTrajectoryState &fts,
                  CLHEP::Hep3Vector &p3,
                  CLHEP::Hep3Vector &r3,
                  int &charge,
                  AlgebraicSymMatrix66 &cov);

  FreeTrajectoryState getFromCLHEP(const CLHEP::Hep3Vector &p3,
                                   const CLHEP::Hep3Vector &r3,
                                   int charge,
                                   const AlgebraicSymMatrix66 &cov,
                                   const MagneticField *field);

  void linearExtrapolation(GlobalPoint initialPosition,
                           GlobalVector initialDirection,
                           float zSurface,
                           std::vector<float> &posZY);
  double extrapolate1D(double initPosition, double initDirection, double parameterOfTheLine);
  double lineParameter(double initZPosition, double destZPosition, double initZDirection);
  bool inSensitiveLocalRegion(double xLocal, double yLocal, int station, int ring);
  bool checkLocal(double yLocal, double yBoundary, int station, int ring);
  void chooseDirection(CLHEP::Hep3Vector &innerPosition, CLHEP::Hep3Vector &outerPosition);
  const Propagator *propagator(std::string propagatorName) const;
  TrajectoryStateOnSurface propagate(FreeTrajectoryState &ftsStart, const BoundPlane &bp);
  bool applyTrigger(edm::Handle<edm::TriggerResults> &hltR, const edm::TriggerNames &triggerNames);
  //bool applyTrigger(void);

  //---- Histograms
  TH1F *DataFlow;
  TH1F *TriggersFired;
  //
  TH1F *ALCTPerEvent;
  TH1F *CLCTPerEvent;
  TH1F *recHitsPerEvent;
  TH1F *segmentsPerEvent;
  //---- Histogram set (stations)...
  struct StationHistos {
    TH1F *segmentChi2_ndf;
    TH1F *hitsInSegment;
    TH1F *AllSegments_eta;
    TH1F *EfficientSegments_eta;
    TH1F *ResidualSegments;
    TH2F *EfficientSegments_XY;
    TH2F *InefficientSegments_XY;
    TH1F *EfficientALCT_momTheta;
    TH1F *InefficientALCT_momTheta;
    TH1F *EfficientCLCT_momPhi;
    TH1F *InefficientCLCT_momPhi;
  } StHist[2][4];
  //---- Histogram set (chambers)...
  struct ChamberHistos {
    TH1F *EfficientRechits_inSegment;
    TH1F *InefficientSingleHits;
    TH1F *digiAppearanceCount;
    TH1F *AllSingleHits;
    TH1F *EfficientRechits_good;
    TH1F *EfficientALCT_dydz;
    TH1F *InefficientALCT_dydz;
    TH1F *EfficientCLCT_dxdz;
    TH1F *InefficientCLCT_dxdz;
    TH1F *EfficientStrips;
    TH1F *StripWiresCorrelations;
    TH1F *NoWires_momTheta;
    TH1F *NoStrips_momPhi;
    TH1F *EfficientWireGroups;
    std::vector<TH1F *> Y_InefficientRecHits_inSegment;
    std::vector<TH1F *> Y_EfficientRecHits_inSegment;
    std::vector<TH1F *> Phi_InefficientRecHits_inSegment;
    std::vector<TH1F *> Phi_EfficientRecHits_inSegment;
    //
    TH1F *SimRechits;
    TH1F *SimSimhits;
    /*
    TH1F * SimRechits_each;
    TH1F * SimSimhits_each;
    */
  } ChHist[2][4][3][LastCh - FirstCh + 1];
};

#endif
