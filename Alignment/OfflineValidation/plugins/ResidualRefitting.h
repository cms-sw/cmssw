#ifndef _ResidualRefitting_h__
#define __ResidualRefitting_h_ (1)

#include <vector>
#include <string>

#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
//#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
//#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
//#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

class TrackerTopology;

class ResidualRefitting : public edm::one::EDAnalyzer<> {
  static const int N_MAX_STORED = 10;
  static const int N_MAX_STORED_HIT = 1000;

  static const int PXB = 1;
  static const int PXF = 2;
  static const int TIB = 3;
  static const int TID = 4;
  static const int TOB = 5;
  static const int TEC = 6;

public:
  //	typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair;
  //	typedef std::vector< ConstTrajTrackPair >  ConstTrajTrackPairCollection;

  typedef struct {
    int evtNum_;
    int runNum_;
  } storage_event;

  ResidualRefitting::storage_event eventInfo_;

  typedef struct StorageMuon {
    int n_;

    int charge_[N_MAX_STORED];
    float pt_[N_MAX_STORED];
    float eta_[N_MAX_STORED];
    float p_[N_MAX_STORED];
    float phi_[N_MAX_STORED];
    int numRecHits_[N_MAX_STORED];
    float chiSq_[N_MAX_STORED];
    float ndf_[N_MAX_STORED];
    float chiSqOvrNdf_[N_MAX_STORED];

    StorageMuon() : n_(0) {
      for (int i = 0; i < N_MAX_STORED; ++i) {
        charge_[i] = 0;
        pt_[i] = 0.;
        eta_[i] = 0.;
        p_[i] = 0.;
        phi_[i] = 0.;
        numRecHits_[i] = 0;
        chiSq_[i] = 0.;
        ndf_[i] = 0.;
        chiSqOvrNdf_[i] = 0.;
      }
    }
  } storage_muon;  // Storage for standard muon information

  typedef struct StorageHit {
    int n_;
    int muonLink_[N_MAX_STORED_HIT];

    int system_[N_MAX_STORED_HIT];
    int endcap_[N_MAX_STORED_HIT];
    int station_[N_MAX_STORED_HIT];
    int ring_[N_MAX_STORED_HIT];
    int chamber_[N_MAX_STORED_HIT];
    int layer_[N_MAX_STORED_HIT];
    int superLayer_[N_MAX_STORED_HIT];
    int wheel_[N_MAX_STORED_HIT];
    int sector_[N_MAX_STORED_HIT];

    float gpX_[N_MAX_STORED_HIT];
    float gpY_[N_MAX_STORED_HIT];
    float gpZ_[N_MAX_STORED_HIT];
    float gpEta_[N_MAX_STORED_HIT];
    float gpPhi_[N_MAX_STORED_HIT];
    float lpX_[N_MAX_STORED_HIT];
    float lpY_[N_MAX_STORED_HIT];
    float lpZ_[N_MAX_STORED_HIT];

    StorageHit() : n_(0) {
      for (int i = 0; i < N_MAX_STORED_HIT; ++i) {
        muonLink_[i] = 0;
        system_[i] = 0;
        endcap_[i] = 0;
        station_[i] = 0;
        ring_[i] = 0;
        chamber_[i] = 0;
        layer_[i] = 0;
        superLayer_[i] = 0;
        wheel_[i] = 0;
        sector_[i] = 0;
        gpX_[i] = 0.;
        gpY_[i] = 0.;
        gpZ_[i] = 0.;
        gpEta_[i] = 0.;
        gpPhi_[i] = 0.;
        lpX_[i] = 0.;
        lpY_[i] = 0.;
        lpZ_[i] = 0.;
      }
    }
  } storage_hit;

  typedef struct StorageTrackExtrap {
    int n_;

    int muonLink_[N_MAX_STORED_HIT];
    int recLink_[N_MAX_STORED_HIT];
    float gpX_[N_MAX_STORED_HIT];
    float gpY_[N_MAX_STORED_HIT];
    float gpZ_[N_MAX_STORED_HIT];
    float gpEta_[N_MAX_STORED_HIT];
    float gpPhi_[N_MAX_STORED_HIT];
    float lpX_[N_MAX_STORED_HIT];
    float lpY_[N_MAX_STORED_HIT];
    float lpZ_[N_MAX_STORED_HIT];
    float resX_[N_MAX_STORED_HIT];
    float resY_[N_MAX_STORED_HIT];
    float resZ_[N_MAX_STORED_HIT];

    StorageTrackExtrap() : n_(0) {
      for (int i = 0; i < N_MAX_STORED_HIT; ++i) {
        muonLink_[i] = 0;
        recLink_[i] = 0;
        gpX_[i] = 0.;
        gpY_[i] = 0.;
        gpZ_[i] = 0.;
        gpEta_[i] = 0.;
        gpPhi_[i] = 0.;
        lpX_[i] = 0.;
        lpY_[i] = 0.;
        lpZ_[i] = 0.;
        resX_[i] = 0.;
        resY_[i] = 0.;
        resZ_[i] = 0.;
      }
    }
  } storage_trackExtrap;

  typedef struct StorageTrackHit {
    int n_;

    int muonLink_[N_MAX_STORED_HIT];
    int detector_[N_MAX_STORED_HIT];
    int subdetector_[N_MAX_STORED_HIT];
    int blade_[N_MAX_STORED_HIT];
    int disk_[N_MAX_STORED_HIT];
    int ladder_[N_MAX_STORED_HIT];
    int layer_[N_MAX_STORED_HIT];
    int module_[N_MAX_STORED_HIT];
    int panel_[N_MAX_STORED_HIT];
    int ring_[N_MAX_STORED_HIT];
    int side_[N_MAX_STORED_HIT];
    int wheel_[N_MAX_STORED_HIT];

    float gpX_[N_MAX_STORED_HIT];
    float gpY_[N_MAX_STORED_HIT];
    float gpZ_[N_MAX_STORED_HIT];
    float gpEta_[N_MAX_STORED_HIT];
    float gpPhi_[N_MAX_STORED_HIT];
    float lpX_[N_MAX_STORED_HIT];
    float lpY_[N_MAX_STORED_HIT];
    float lpZ_[N_MAX_STORED_HIT];

    StorageTrackHit() : n_(0) {
      for (int i = 0; i < N_MAX_STORED_HIT; ++i) {
        muonLink_[i] = 0;
        detector_[i] = 0;
        subdetector_[i] = 0;
        blade_[i] = 0;
        disk_[i] = 0;
        ladder_[i] = 0;
        layer_[i] = 0;
        module_[i] = 0;
        panel_[i] = 0;
        ring_[i] = 0;
        side_[i] = 0;
        wheel_[i] = 0;
        gpX_[i] = 0.;
        gpY_[i] = 0.;
        gpZ_[i] = 0.;
        gpEta_[i] = 0.;
        gpPhi_[i] = 0.;
        lpX_[i] = 0.;
        lpY_[i] = 0.;
        lpZ_[i] = 0.;
      }
    }
  } storage_trackHit;

  //Standard Muon info storage
  ResidualRefitting::storage_muon storageGmrOld_, storageGmrNew_, storageSamNew_, storageTrkNew_, storageGmrNoSt1_,
      storageSamNoSt1_, storageGmrNoSt2_, storageSamNoSt2_, storageGmrNoSt3_, storageSamNoSt3_, storageGmrNoSt4_,
      storageSamNoSt4_,

      storageGmrNoPXBLayer1, storageGmrNoPXBLayer2, storageGmrNoPXBLayer3, storageTrkNoPXBLayer1, storageTrkNoPXBLayer2,
      storageTrkNoPXBLayer3,

      storageGmrNoPXF, storageTrkNoPXF,

      storageGmrNoTIBLayer1, storageGmrNoTIBLayer2, storageGmrNoTIBLayer3, storageGmrNoTIBLayer4, storageTrkNoTIBLayer1,
      storageTrkNoTIBLayer2, storageTrkNoTIBLayer3, storageTrkNoTIBLayer4, storageGmrNoTID, storageTrkNoTID,
      storageGmrNoTOBLayer1, storageGmrNoTOBLayer2, storageGmrNoTOBLayer3, storageGmrNoTOBLayer4, storageGmrNoTOBLayer5,
      storageGmrNoTOBLayer6, storageTrkNoTOBLayer1, storageTrkNoTOBLayer2, storageTrkNoTOBLayer3, storageTrkNoTOBLayer4,
      storageTrkNoTOBLayer5, storageTrkNoTOBLayer6, storageGmrNoTEC, storageTrkNoTEC;

  //Rec hit storage
  ResidualRefitting::storage_hit storageRecMuon_;
  ResidualRefitting::storage_trackHit storageTrackHit_;

  //Track Extrapolation to Muon System
  ResidualRefitting::storage_trackExtrap storageTrackExtrapRec_, storageTrackExtrapRecNoSt1_,
      storageTrackExtrapRecNoSt2_, storageTrackExtrapRecNoSt3_, storageTrackExtrapRecNoSt4_;

  //Track Extrapolation with Cylinder
  ResidualRefitting::storage_trackExtrap trackExtrap120_, samExtrap120_;

  //Track Extrapolation to Tracker system
  ResidualRefitting::storage_trackExtrap storageTrackExtrapTracker_, storageTrackNoPXBLayer1, storageTrackNoPXBLayer2,
      storageTrackNoPXBLayer3, storageTrackNoPXF, storageTrackNoTIBLayer1, storageTrackNoTIBLayer2,
      storageTrackNoTIBLayer3, storageTrackNoTIBLayer4, storageTrackNoTID, storageTrackNoTOBLayer1,
      storageTrackNoTOBLayer2, storageTrackNoTOBLayer3, storageTrackNoTOBLayer4, storageTrackNoTOBLayer5,
      storageTrackNoTOBLayer6, storageTrackNoTEC;

  //
  // Start of the method declarations
  //

  explicit ResidualRefitting(const edm::ParameterSet&);
  ~ResidualRefitting() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;
  //Zero Storage
  void zero_storage();
  void zero_muon(ResidualRefitting::storage_muon* str);
  void zero_trackExtrap(ResidualRefitting::storage_trackExtrap* str);
  void branchMuon(ResidualRefitting::storage_muon& storageTmp, std::string branchName);
  void branchTrackExtrap(ResidualRefitting::storage_trackExtrap& storageTmp, std::string branchName);

  //	void collectTrackRecExtrap(reco::MuonCollection::const_iterator muon, ResidualRefitting::storage_trackExtrap& storeTemp);
  void muonInfo(ResidualRefitting::storage_muon& storeMuon, reco::TrackRef muon, int val);

  void CollectTrackHits(edm::Handle<reco::TrackCollection> trackColl,
                        ResidualRefitting::storage_trackExtrap& trackExtrap,
                        const edm::EventSetup& eventSetup);
  void StoreTrackerRecHits(DetId detid, const TrackerTopology* tTopo, int iTrack, int iRec);
  void NewTrackMeasurements(edm::Handle<reco::TrackCollection> trackCollOrig,
                            edm::Handle<reco::TrackCollection> trackColl,
                            ResidualRefitting::storage_trackExtrap& trackExtrap);
  int MatchTrackWithRecHits(reco::TrackCollection::const_iterator trackIt, edm::Handle<reco::TrackCollection> ref);

  bool IsSameHit(TrackingRecHit const& hit1, TrackingRecHit const& hit2);

  void trkExtrap(const DetId& detid,
                 int iTrkLink,
                 int iTrk,
                 int iRec,
                 const FreeTrajectoryState& freeTrajState,
                 const LocalPoint& recPoint,
                 storage_trackExtrap& storeTemp);

  void cylExtrapTrkSam(int recNum, reco::TrackRef track, ResidualRefitting::storage_trackExtrap& storage, double rho);

  //Simplifiying functions
  FreeTrajectoryState freeTrajStateMuon(reco::TrackRef muon);  //Returns a Free Trajectory State
                                                               //Debug Data Dumps
  //	void dumpRecoMuonColl(reco::MuonCollection::const_iterator muon); //
  //	void dumpRecoTrack(reco::TrackCollection::const_iterator muon);
  void dumpTrackRef(reco::TrackRef muon, std::string str);
  void dumpTrackExtrap(const ResidualRefitting::storage_trackExtrap& track);
  void dumpTrackHits(const ResidualRefitting::storage_trackHit& hit);
  void dumpMuonRecHits(const ResidualRefitting::storage_hit& hit);

  int ReturnSector(DetId detid);
  int ReturnStation(DetId detid);

  // Deprecated Functions
  void omitStation(edm::Handle<reco::MuonCollection> funcMuons,
                   edm::Handle<reco::TrackCollection>,
                   ResidualRefitting::storage_muon& storeGmr,
                   ResidualRefitting::storage_muon& storeSam,
                   ResidualRefitting::storage_trackExtrap& storeExtrap,
                   int omitStation);
  void omitTrackerSystem(edm::Handle<reco::MuonCollection> trkMuons,
                         ResidualRefitting::storage_muon& storeGmr,
                         ResidualRefitting::storage_muon& storeTrk,
                         ResidualRefitting::storage_trackExtrap& storeExtrap,
                         int omitSystem);

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> trackingGeometryToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;

  // output histogram file name
  std::string outputFileName_;
  //edm::InputTag PropagatorSource_;
  std::string PropagatorSource_;

  // names of product labels
  edm::InputTag tracks_, muons_, muonsRemake_, muonsNoStation1_, muonsNoStation2_, muonsNoStation3_,
      muonsNoStation4_,  //Global Muon Collections
      muonsNoPXBLayer1_, muonsNoPXBLayer2_, muonsNoPXBLayer3_, muonsNoPXF_, muonsNoTIBLayer1_, muonsNoTIBLayer2_,
      muonsNoTIBLayer3_, muonsNoTIBLayer4_, muonsNoTID_, muonsNoTOBLayer1_, muonsNoTOBLayer2_, muonsNoTOBLayer3_,
      muonsNoTOBLayer4_, muonsNoTOBLayer5_, muonsNoTOBLayer6_, muonsNoTEC_;
  //	   tjTag;

  bool debug_;

  // output ROOT file
  TFile* outputFile_;

  TTree* outputTree_;
  TBranch* outputBranch_;

  //	unsigned int nBins_;

  const MagneticField* theField;
  const edm::ESHandle<GlobalTrackingGeometry> trackingGeometry;
  MuonServiceProxy* theService;
  edm::ESHandle<Propagator> thePropagator;
};

#endif
