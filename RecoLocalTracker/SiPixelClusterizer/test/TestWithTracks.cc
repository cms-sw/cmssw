// File: ReadPixClusters.cc
// Description: TO test the pixel clusters with tracks (full)
// Author: Danek Kotlinski
// Creation Date:  Initial version. 3/06
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

//#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//#include "Geometry/CommonTopologies/interface/PixelTopology.h"

// For L1
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// For HLT
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Common/interface/TriggerNames.h"

// For tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

//#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
//#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
//#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

//#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"

//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
//#include "TrackingTools/PatternTools/interface/Trajectory.h"
//#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include <DataFormats/VertexReco/interface/VertexFwd.h>

// For luminisoty
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"

// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TProfile.h>
#include <TVector3.h>

#define HISTOS
//#define L1
//#define HLT

#define VDM_STUDIES

const bool isData = false;  // set false for MC

using namespace std;

class TestWithTracks : public edm::EDAnalyzer {
public:
  explicit TestWithTracks(const edm::ParameterSet &conf);
  virtual ~TestWithTracks();
  virtual void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  virtual void beginRun(edm::Run const &, edm::EventSetup const &) override;
  virtual void beginJob() override;
  virtual void endJob() override;

private:
  edm::EDGetTokenT<LumiSummary> lumiToken_;
  edm::EDGetTokenT<edm::ConditionsInLumiBlock> condToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> l1gtrrToken_;
  edm::EDGetTokenT<edm::TriggerResults> hltToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<reco::TrackCollection> srcToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trackAssocToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  //const static bool PRINT = false;
  bool PRINT;
  float countTracks, countGoodTracks, countTracksInPix, countPVs, countEvents, countLumi;

  TH1D *hcharge1, *hcharge2, *hcharge3, *hcharge4, *hcharge5;  // ADD FPIX
  TH1D *hsize1, *hsize2, *hsize3, *hsizex1, *hsizex2, *hsizex3, *hsizey1, *hsizey2, *hsizey3;
  TH1D *hsize4, *hsize5,  // ADD FPIX
      *hsizex4, *hsizex5, *hsizey4, *hsizey5;

  TH1D *hclusPerTrk1, *hclusPerTrk2, *hclusPerTrk3, *hclusPerTrk4, *hclusPerTrk5;
  TH1D *hclusPerLay1, *hclusPerLay2, *hclusPerLay3;
  TH1D *hclusPerDisk1, *hclusPerDisk2, *hclusPerDisk3, *hclusPerDisk4;
  TH1D *hclusPerTrk, *hclusPerTrkB, *hclusPerTrkF;

  TH2F *hDetMap1, *hDetMap2, *hDetMap3;           // clusters
  TH2F *hcluDetMap1, *hcluDetMap2, *hcluDetMap3;  // MODULE PROJECTION

  TH2F *hpvxy, *hclusMap1, *hclusMap2, *hclusMap3;  // Z vs PHI

  TH1D *hpvz, *hpvr, *hNumPv, *hNumPvClean;
  TH1D *hPt, *hEta, *hDz, *hD0, *hzdiff;

  TH1D *hl1a, *hl1t, *hlt1;

  TH1D *hclusBpix, *hpixBpix;
  TH1D *htracks, *htracksGood, *htracksGoodInPix;

  TProfile *hclumult1, *hclumult2, *hclumult3;
  TProfile *hclumultx1, *hclumultx2, *hclumultx3;
  TProfile *hclumulty1, *hclumulty2, *hclumulty3;
  TProfile *hcluchar1, *hcluchar2, *hcluchar3;

  TProfile *hclumultld1, *hclumultld2, *hclumultld3;
  TProfile *hclumultxld1, *hclumultxld2, *hclumultxld3;
  TProfile *hclumultyld1, *hclumultyld2, *hclumultyld3;
  TProfile *hclucharld1, *hclucharld2, *hclucharld3;

  TProfile *htracksls, *hpvsls, *htrackslsn, *hpvslsn, *hintgl, *hinstl, *hbeam1, *hbeam2;
  TProfile *hmult1, *hmult2, *hmult3;
  TProfile *hclusPerTrkVsEta, *hclusPerTrkVsPt, *hclusPerTrkVsls, *hclusPerTrkVsEtaB, *hclusPerTrkVsEtaF;

  TH1D *hlumi, *hlumi0, *hbx, *hbx0;

  TH1D *recHitXError1, *recHitXError2, *recHitXError3;
  TH1D *recHitYError1, *recHitYError2, *recHitYError3;
  TH1D *recHitXAlignError1, *recHitXAlignError2, *recHitXAlignError3;
  TH1D *recHitYAlignError1, *recHitYAlignError2, *recHitYAlignError3;
  TH1D *recHitXError4, *recHitXError5, *recHitXError6, *recHitXError7;
  TH1D *recHitYError4, *recHitYError5, *recHitYError6, *recHitYError7;
  TH1D *recHitXAlignError4, *recHitXAlignError5, *recHitXAlignError6, *recHitXAlignError7;
  TH1D *recHitYAlignError4, *recHitYAlignError5, *recHitYAlignError6, *recHitYAlignError7;
  TProfile *hErrorXB, *hErrorYB, *hErrorXF, *hErrorYF;
  TProfile *hAErrorXB, *hAErrorYB, *hAErrorXF, *hAErrorYF;

#ifdef VDM_STUDIES
  TProfile *hcharCluls, *hcharPixls, *hsizeCluls, *hsizeXCluls;
  TProfile *hcharCluls1, *hcharPixls1, *hsizeCluls1, *hsizeXCluls1;
  TProfile *hcharCluls2, *hcharPixls2, *hsizeCluls2, *hsizeXCluls2;
  TProfile *hcharCluls3, *hcharPixls3, *hsizeCluls3, *hsizeXCluls3;
  TProfile *hclusls;  //   *hpixls;
  TProfile *hclusls1, *hclusls2, *hclusls3;
  TProfile *hclubx, *hpvbx, *htrackbx;  // *hcharClubx, *hcharPixbx,*hsizeClubx, *hsizeYClubx;
#endif
};
/////////////////////////////////////////////////////////////////
// Contructor,
TestWithTracks::TestWithTracks(edm::ParameterSet const &conf) {
  PRINT = conf.getUntrackedParameter<bool>("Verbosity", false);
  lumiToken_ = consumes<LumiSummary>(edm::InputTag("lumiProducer"));
  condToken_ = consumes<edm::ConditionsInLumiBlock>(edm::InputTag("conditionsInEdm"));
  l1gtrrToken_ = consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag("gtDigis"));
  hltToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", "HLT"));
  vtxToken_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  srcToken_ = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("src"));
  trackAssocToken_ =
      consumes<TrajTrackAssociationCollection>(edm::InputTag(conf.getParameter<std::string>("trajectoryInput")));
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  //if(PRINT) cout<<" Construct "<<endl;
}

// Virtual destructor needed.
TestWithTracks::~TestWithTracks() {}

// ------------ method called at the begining   ------------
void TestWithTracks::beginRun(edm::Run const &, const edm::EventSetup &iSetup) {
  cout << "BeginRun, Verbosity =  " << PRINT << endl;
}

// ------------ method called at the begining   ------------
void TestWithTracks::beginJob() {
  cout << "BeginJob, Verbosity " << PRINT << endl;

  countTracks = 0.;
  countGoodTracks = 0.;
  countTracksInPix = 0.;
  countPVs = 0.;
  countEvents = 0.;
  countLumi = 0.;

#ifdef HISTOS
  edm::Service<TFileService> fs;

  int sizeH = 20;
  float lowH = -0.5;
  float highH = 19.5;
  hclusPerTrk1 = fs->make<TH1D>("hclusPerTrk1", "Clus per track l1", sizeH, lowH, highH);
  hclusPerTrk2 = fs->make<TH1D>("hclusPerTrk2", "Clus per track l2", sizeH, lowH, highH);
  hclusPerTrk3 = fs->make<TH1D>("hclusPerTrk3", "Clus per track l3", sizeH, lowH, highH);
  hclusPerTrk4 = fs->make<TH1D>("hclusPerTrk4", "Clus per track d1", sizeH, lowH, highH);
  hclusPerTrk5 = fs->make<TH1D>("hclusPerTrk5", "Clus per track d2", sizeH, lowH, highH);
  hclusPerTrk = fs->make<TH1D>("hclusPerTrk", "Clus per track", sizeH, lowH, highH);
  hclusPerTrkB = fs->make<TH1D>("hclusPerTrkB", "B Clus per track", sizeH, lowH, highH);
  hclusPerTrkF = fs->make<TH1D>("hclusPerTrkF", "F Clus per track", sizeH, lowH, highH);

  sizeH = 2000;
  highH = 1999.5;
  hclusPerLay1 = fs->make<TH1D>("hclusPerLay1", "Clus per layer l1", sizeH, lowH, highH);
  hclusPerLay2 = fs->make<TH1D>("hclusPerLay2", "Clus per layer l2", sizeH, lowH, highH);
  hclusPerLay3 = fs->make<TH1D>("hclusPerLay3", "Clus per layer l3", sizeH, lowH, highH);
  hclusPerDisk1 = fs->make<TH1D>("hclusPerDisk1", "Clus per disk 1", sizeH, lowH, highH);
  hclusPerDisk2 = fs->make<TH1D>("hclusPerDisk2", "Clus per disk 2", sizeH, lowH, highH);
  hclusPerDisk3 = fs->make<TH1D>("hclusPerDisk3", "Clus per disk 3", sizeH, lowH, highH);
  hclusPerDisk4 = fs->make<TH1D>("hclusPerDisk4", "Clus per disk 4", sizeH, lowH, highH);

  hcharge1 = fs->make<TH1D>("hcharge1", "Clu charge l1", 400, 0., 200.);  //in ke
  hcharge2 = fs->make<TH1D>("hcharge2", "Clu charge l2", 400, 0., 200.);
  hcharge3 = fs->make<TH1D>("hcharge3", "Clu charge l3", 400, 0., 200.);
  hcharge4 = fs->make<TH1D>("hcharge4", "Clu charge d1", 400, 0., 200.);
  hcharge5 = fs->make<TH1D>("hcharge5", "Clu charge d2", 400, 0., 200.);

  hsize1 = fs->make<TH1D>("hsize1", "layer 1 clu size", 100, -0.5, 99.5);
  hsize2 = fs->make<TH1D>("hsize2", "layer 2 clu size", 100, -0.5, 99.5);
  hsize3 = fs->make<TH1D>("hsize3", "layer 3 clu size", 100, -0.5, 99.5);
  hsizex1 = fs->make<TH1D>("hsizex1", "lay1 clu size in x", 20, -0.5, 19.5);
  hsizex2 = fs->make<TH1D>("hsizex2", "lay2 clu size in x", 20, -0.5, 19.5);
  hsizex3 = fs->make<TH1D>("hsizex3", "lay3 clu size in x", 20, -0.5, 19.5);
  hsizey1 = fs->make<TH1D>("hsizey1", "lay1 clu size in y", 30, -0.5, 29.5);
  hsizey2 = fs->make<TH1D>("hsizey2", "lay2 clu size in y", 30, -0.5, 29.5);
  hsizey3 = fs->make<TH1D>("hsizey3", "lay3 clu size in y", 30, -0.5, 29.5);

  hsize4 = fs->make<TH1D>("hsize4", "disk 1 clu size", 100, -0.5, 99.5);
  hsize5 = fs->make<TH1D>("hsize5", "disk 2 clu size", 100, -0.5, 99.5);
  hsizex4 = fs->make<TH1D>("hsizex4", "d1 clu size in x", 20, -0.5, 19.5);
  hsizex5 = fs->make<TH1D>("hsizex5", "d2 clu size in x", 20, -0.5, 19.5);
  hsizey4 = fs->make<TH1D>("hsizey4", "d1 clu size in y", 30, -0.5, 29.5);
  hsizey5 = fs->make<TH1D>("hsizey5", "d2 clu size in y", 30, -0.5, 29.5);

  hDetMap1 = fs->make<TH2F>("hDetMap1", "layer 1 clus map", 9, 0., 9., 23, 0., 23.);
  hDetMap2 = fs->make<TH2F>("hDetMap2", "layer 2 clus map", 9, 0., 9., 33, 0., 33.);
  hDetMap3 = fs->make<TH2F>("hDetMap3", "layer 3 clus map", 9, 0., 9., 45, 0., 45.);
  //hpixDetMap1 = fs->make<TH2F>( "hpixDetMap1", "pix det layer 1",
  //	      416,0.,416.,160,0.,160.);
  //hpixDetMap2 = fs->make<TH2F>( "hpixDetMap2", "pix det layer 2",
  //	      416,0.,416.,160,0.,160.);
  //hpixDetMap3 = fs->make<TH2F>( "hpixDetMap3", "pix det layer 3",
  //	      416,0.,416.,160,0.,160.);
  hcluDetMap1 = fs->make<TH2F>("hcluDetMap1", "clu det layer 1", 416, 0., 416., 160, 0., 160.);
  hcluDetMap2 = fs->make<TH2F>("hcluDetMap2", "clu det layer 2", 416, 0., 416., 160, 0., 160.);
  hcluDetMap3 = fs->make<TH2F>("hcluDetMap3", "clu det layer 3", 416, 0., 416., 160, 0., 160.);

  htracksGoodInPix = fs->make<TH1D>("htracksGoodInPix", "count good tracks in pix", 2000, -0.5, 1999.5);
  htracksGood = fs->make<TH1D>("htracksGood", "count good tracks", 2000, -0.5, 1999.5);
  htracks = fs->make<TH1D>("htracks", "count tracks", 2000, -0.5, 1999.5);
  hclusBpix = fs->make<TH1D>("hclusBpix", "count clus in bpix", 200, -0.5, 1999.5);
  hpixBpix = fs->make<TH1D>("hpixBpix", "count pixels", 200, -0.5, 1999.5);

  hpvxy = fs->make<TH2F>("hpvxy", "pv xy", 100, -1., 1., 100, -1., 1.);
  hpvz = fs->make<TH1D>("hpvz", "pv z", 1000, -50., 50.);
  hpvr = fs->make<TH1D>("hpvr", "pv r", 100, 0., 1.);
  hNumPv = fs->make<TH1D>("hNumPv", "num of pv", 100, 0., 100.);
  hNumPvClean = fs->make<TH1D>("hNumPvClean", "num of pv clean", 100, 0., 100.);

  hPt = fs->make<TH1D>("hPt", "pt", 120, 0., 120.);
  hEta = fs->make<TH1D>("hEta", "eta", 50, -2.5, 2.5);
  hD0 = fs->make<TH1D>("hD0", "d0", 500, 0., 5.);
  hDz = fs->make<TH1D>("hDz", "pt", 250, -25., 25.);
  hzdiff = fs->make<TH1D>("hzdiff", "PVz-Trackz", 200, -10., 10.);

  hl1a = fs->make<TH1D>("hl1a", "l1a", 128, -0.5, 127.5);
  hl1t = fs->make<TH1D>("hl1t", "l1t", 128, -0.5, 127.5);
  hlt1 = fs->make<TH1D>("hlt1", "hlt1", 256, -0.5, 255.5);

  hclumult1 = fs->make<TProfile>("hclumult1", "cluster size layer 1", 60, -3., 3., 0.0, 100.);
  hclumult2 = fs->make<TProfile>("hclumult2", "cluster size layer 2", 60, -3., 3., 0.0, 100.);
  hclumult3 = fs->make<TProfile>("hclumult3", "cluster size layer 3", 60, -3., 3., 0.0, 100.);

  hclumultx1 = fs->make<TProfile>("hclumultx1", "cluster x-size layer 1", 60, -3., 3., 0.0, 100.);
  hclumultx2 = fs->make<TProfile>("hclumultx2", "cluster x-size layer 2", 60, -3., 3., 0.0, 100.);
  hclumultx3 = fs->make<TProfile>("hclumultx3", "cluster x-size layer 3", 60, -3., 3., 0.0, 100.);

  hclumulty1 = fs->make<TProfile>("hclumulty1", "cluster y-size layer 1", 60, -3., 3., 0.0, 100.);
  hclumulty2 = fs->make<TProfile>("hclumulty2", "cluster y-size layer 2", 60, -3., 3., 0.0, 100.);
  hclumulty3 = fs->make<TProfile>("hclumulty3", "cluster y-size layer 3", 60, -3., 3., 0.0, 100.);

  hcluchar1 = fs->make<TProfile>("hcluchar1", "cluster char layer 1", 60, -3., 3., 0.0, 1000.);
  hcluchar2 = fs->make<TProfile>("hcluchar2", "cluster char layer 2", 60, -3., 3., 0.0, 1000.);
  hcluchar3 = fs->make<TProfile>("hcluchar3", "cluster char layer 3", 60, -3., 3., 0.0, 1000.);

  // profiles versus ladder
  hclumultld1 = fs->make<TProfile>("hclumultld1", "cluster size layer 1", 23, -11.5, 11.5, 0.0, 100.);
  hclumultld2 = fs->make<TProfile>("hclumultld2", "cluster size layer 2", 35, -17.5, 17.5, 0.0, 100.);
  hclumultld3 = fs->make<TProfile>("hclumultld3", "cluster size layer 3", 47, -23.5, 23.5, 0.0, 100.);

  hclumultxld1 = fs->make<TProfile>("hclumultxld1", "cluster x-size layer 1", 23, -11.5, 11.5, 0.0, 100.);
  hclumultxld2 = fs->make<TProfile>("hclumultxld2", "cluster x-size layer 2", 35, -17.5, 17.5, 0.0, 100.);
  hclumultxld3 = fs->make<TProfile>("hclumultxld3", "cluster x-size layer 3", 47, -23.5, 23.5, 0.0, 100.);

  hclumultyld1 = fs->make<TProfile>("hclumultyld1", "cluster y-size layer 1", 23, -11.5, 11.5, 0.0, 100.);
  hclumultyld2 = fs->make<TProfile>("hclumultyld2", "cluster y-size layer 2", 35, -17.5, 17.5, 0.0, 100.);
  hclumultyld3 = fs->make<TProfile>("hclumultyld3", "cluster y-size layer 3", 47, -23.5, 23.5, 0.0, 100.);

  hclucharld1 = fs->make<TProfile>("hclucharld1", "cluster char layer 1", 23, -11.5, 11.5, 0.0, 1000.);
  hclucharld2 = fs->make<TProfile>("hclucharld2", "cluster char layer 2", 35, -17.5, 17.5, 0.0, 1000.);
  hclucharld3 = fs->make<TProfile>("hclucharld3", "cluster char layer 3", 47, -23.5, 23.5, 0.0, 1000.);

  hintgl = fs->make<TProfile>("hintgl", "inst lumi vs ls ", 1000, 0., 3000., 0.0, 10000.);
  hinstl = fs->make<TProfile>("hinstl", "intg lumi vs ls ", 1000, 0., 3000., 0.0, 100.);
  hbeam1 = fs->make<TProfile>("hbeam1", "beam1 vs ls ", 1000, 0., 3000., 0.0, 1000.);
  hbeam2 = fs->make<TProfile>("hbeam2", "beam2 vs ls ", 1000, 0., 3000., 0.0, 1000.);

  htracksls = fs->make<TProfile>("htracksls", "tracks with pix hits  vs ls", 1000, 0., 3000., 0.0, 10000.);
  hpvsls = fs->make<TProfile>("hpvsls", "pvs  vs ls", 1000, 0., 3000., 0.0, 1000.);
  htrackslsn = fs->make<TProfile>("htrackslsn", "tracks with pix hits/lumi  vs ls", 1000, 0., 3000., 0.0, 10000.);
  hpvslsn = fs->make<TProfile>("hpvslsn", "pvs/lumi  vs ls", 1000, 0., 3000., 0.0, 1000.);

  hmult1 = fs->make<TProfile>("hmult1", "clu mult layer 1", 10, 0., 10., 0.0, 1000.);
  hmult2 = fs->make<TProfile>("hmult2", "clu mult layer 2", 10, 0., 10., 0.0, 1000.);
  hmult3 = fs->make<TProfile>("hmult3", "clu mult layer 3", 10, 0., 10., 0.0, 1000.);

  hclusPerTrkVsEta = fs->make<TProfile>("hclusPerTrkVsEta", "clus per trk vs.eta", 60, -3., 3., 0.0, 100.);
  hclusPerTrkVsPt = fs->make<TProfile>("hclusPerTrkVsPt", "clus per trk vs.pt", 120, 0., 120., 0.0, 100.);
  hclusPerTrkVsls = fs->make<TProfile>("hclusPerTrkVsls", "clus per trk vs.ls", 300, 0., 3000., 0.0, 100.);
  hclusPerTrkVsEtaF = fs->make<TProfile>("hclusPerTrkVsEtaF", "F clus per trk vs.eta", 60, -3., 3., 0.0, 100.);
  hclusPerTrkVsEtaB = fs->make<TProfile>("hclusPerTrkVsEtaB", "B clus per trk vs.eta", 60, -3., 3., 0.0, 100.);

  hlumi0 = fs->make<TH1D>("hlumi0", "lumi", 2000, 0, 2000.);
  hlumi = fs->make<TH1D>("hlumi", "lumi", 2000, 0, 2000.);
  hbx0 = fs->make<TH1D>("hbx0", "bx", 4000, 0, 4000.);
  hbx = fs->make<TH1D>("hbx", "bx", 4000, 0, 4000.);

  hclusMap1 = fs->make<TH2F>("hclusMap1", "clus - lay1", 260, -26., 26., 350, -3.5, 3.5);
  hclusMap2 = fs->make<TH2F>("hclusMap2", "clus - lay2", 260, -26., 26., 350, -3.5, 3.5);
  hclusMap3 = fs->make<TH2F>("hclusMap3", "clus - lay3", 260, -26., 26., 350, -3.5, 3.5);

  // RecHit errors
  // alignment errors
  recHitXAlignError1 = fs->make<TH1D>("recHitXAlignError1", "RecHit X Alignment errors bpix 1", 100, 0., 100.);
  recHitYAlignError1 = fs->make<TH1D>("recHitYAlignError1", "RecHit Y Alignment errors bpix 1", 100, 0., 100.);
  recHitXAlignError2 = fs->make<TH1D>("recHitXAlignError2", "RecHit X Alignment errors bpix 2", 100, 0., 100.);
  recHitYAlignError2 = fs->make<TH1D>("recHitYAlignError2", "RecHit Y Alignment errors bpix 2", 100, 0., 100.);
  recHitXAlignError3 = fs->make<TH1D>("recHitXAlignError3", "RecHit X Alignment errors bpix 3", 100, 0., 100.);
  recHitYAlignError3 = fs->make<TH1D>("recHitYAlignError3", "RecHit Y Alignment errors bpix 3", 100, 0., 100.);
  recHitXAlignError4 = fs->make<TH1D>("recHitXAlignError4", "RecHit X Alignment errors fpix -d2", 100, 0., 100.);
  recHitYAlignError4 = fs->make<TH1D>("recHitYAlignError4", "RecHit Y Alignment errors fpix -d2", 100, 0., 100.);
  recHitXAlignError5 = fs->make<TH1D>("recHitXAlignError5", "RecHit X Alignment errors fpix -d1", 100, 0., 100.);
  recHitYAlignError5 = fs->make<TH1D>("recHitYAlignError5", "RecHit Y Alignment errors fpix -d1", 100, 0., 100.);
  recHitXAlignError6 = fs->make<TH1D>("recHitXAlignError6", "RecHit X Alignment errors fpix +d1", 100, 0., 100.);
  recHitYAlignError6 = fs->make<TH1D>("recHitYAlignError6", "RecHit Y Alignment errors fpix +d1", 100, 0., 100.);
  recHitXAlignError7 = fs->make<TH1D>("recHitXAlignError7", "RecHit X Alignment errors fpix +d2", 100, 0., 100.);
  recHitYAlignError7 = fs->make<TH1D>("recHitYAlignError7", "RecHit Y Alignment errors fpix +d2", 100, 0., 100.);

  recHitXError1 = fs->make<TH1D>("recHitXError1", "RecHit X errors bpix 1", 100, 0., 100.);
  recHitYError1 = fs->make<TH1D>("recHitYError1", "RecHit Y errors bpix 1", 100, 0., 100.);
  recHitXError2 = fs->make<TH1D>("recHitXError2", "RecHit X errors bpix 2", 100, 0., 100.);
  recHitYError2 = fs->make<TH1D>("recHitYError2", "RecHit Y errors bpix 2", 100, 0., 100.);
  recHitXError3 = fs->make<TH1D>("recHitXError3", "RecHit X errors bpix 3", 100, 0., 100.);
  recHitYError3 = fs->make<TH1D>("recHitYError3", "RecHit Y errors bpix 3", 100, 0., 100.);
  recHitXError4 = fs->make<TH1D>("recHitXError4", "RecHit X errors fpix -d2", 100, 0., 100.);
  recHitYError4 = fs->make<TH1D>("recHitYError4", "RecHit Y errors fpix -d2", 100, 0., 100.);
  recHitXError5 = fs->make<TH1D>("recHitXError5", "RecHit X errors fpix -d1", 100, 0., 100.);
  recHitYError5 = fs->make<TH1D>("recHitYError5", "RecHit Y errors fpix -d1", 100, 0., 100.);
  recHitXError6 = fs->make<TH1D>("recHitXError6", "RecHit X errors fpix +d1", 100, 0., 100.);
  recHitYError6 = fs->make<TH1D>("recHitYError6", "RecHit Y errors fpix +d1", 100, 0., 100.);
  recHitXError7 = fs->make<TH1D>("recHitXError7", "RecHit X errors fpix +d2", 100, 0., 100.);
  recHitYError7 = fs->make<TH1D>("recHitYError7", "RecHit Y errors fpix +d2", 100, 0., 100.);

  hErrorXB = fs->make<TProfile>("hErrorXB", "bpix x errors per ladder", 220, 0., 220., 0.0, 1000.);
  hErrorXF = fs->make<TProfile>("hErrorXF", "fpix x errors per ladder", 100, 0., 100., 0.0, 1000.);
  hErrorYB = fs->make<TProfile>("hErrorYB", "bpix y errors per ladder", 220, 0., 220., 0.0, 1000.);
  hErrorYF = fs->make<TProfile>("hErrorYF", "fpix y errors per ladder", 100, 0., 100., 0.0, 1000.);

  hAErrorXB = fs->make<TProfile>("hAErrorXB", "bpix x errors per ladder", 220, 0., 220., 0.0, 1000.);
  hAErrorXF = fs->make<TProfile>("hAErrorXF", "fpix x errors per ladder", 100, 0., 100., 0.0, 1000.);
  hAErrorYB = fs->make<TProfile>("hAErrorYB", "bpix y errors per ladder", 220, 0., 220., 0.0, 1000.);
  hAErrorYF = fs->make<TProfile>("hAErrorYF", "fpix y errors per ladder", 100, 0., 100., 0.0, 1000.);

#ifdef VDM_STUDIES
  highH = 3000.;
  sizeH = 1000;

  hclusls = fs->make<TProfile>("hclusls", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  //hpixls  = fs->make<TProfile>("hpixls", "pix vs ls ",sizeH,0.,highH,0.0,100000.);

  hcharCluls = fs->make<TProfile>("hcharCluls", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  //hcharPixls = fs->make<TProfile>("hcharPixls","pix char vs ls",sizeH,0.,highH,0.0,100.);
  hsizeCluls = fs->make<TProfile>("hsizeCluls", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls = fs->make<TProfile>("hsizeXCluls", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);

  hcharCluls1 = fs->make<TProfile>("hcharCluls1", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  //hcharPixls1 = fs->make<TProfile>("hcharPixls1","pix char vs ls",sizeH,0.,highH,0.0,100.);
  hsizeCluls1 = fs->make<TProfile>("hsizeCluls1", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls1 = fs->make<TProfile>("hsizeXCluls1", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharCluls2 = fs->make<TProfile>("hcharCluls2", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  //hcharPixls2 = fs->make<TProfile>("hcharPixls2","pix char vs ls",sizeH,0.,highH,0.0,100.);
  hsizeCluls2 = fs->make<TProfile>("hsizeCluls2", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls2 = fs->make<TProfile>("hsizeXCluls2", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);
  hcharCluls3 = fs->make<TProfile>("hcharCluls3", "clu char vs ls", sizeH, 0., highH, 0.0, 100.);
  //hcharPixls3 = fs->make<TProfile>("hcharPixls3","pix char vs ls",sizeH,0.,highH,0.0,100.);
  hsizeCluls3 = fs->make<TProfile>("hsizeCluls3", "clu size vs ls", sizeH, 0., highH, 0.0, 1000.);
  hsizeXCluls3 = fs->make<TProfile>("hsizeXCluls3", "clu size-x vs ls", sizeH, 0., highH, 0.0, 100.);
  hclusls1 = fs->make<TProfile>("hclusls1", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  //hpixls1  = fs->make<TProfile>("hpixls1", "pix vs ls ",sizeH,0.,highH,0.0,100000.);
  hclusls2 = fs->make<TProfile>("hclusls2", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  //hpixls2  = fs->make<TProfile>("hpixls2", "pix vs ls ",sizeH,0.,highH,0.0,100000.);
  hclusls3 = fs->make<TProfile>("hclusls3", "clus vs ls", sizeH, 0., highH, 0.0, 30000.);
  //hpixls3  = fs->make<TProfile>("hpixls3", "pix vs ls ",sizeH,0.,highH,0.0,100000.);

  // Profiles versus bx
  //hpixbx  = fs->make<TProfile>("hpixbx", "pixs vs bx ",4000,-0.5,3999.5,0.0,1000000.);
  hclubx = fs->make<TProfile>("hclubx", "clus vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
  hpvbx = fs->make<TProfile>("hpvbx", "pvs vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);
  htrackbx = fs->make<TProfile>("htrackbx", "tracks vs bx ", 4000, -0.5, 3999.5, 0.0, 1000000.);

#endif

#endif
}
// ------------ method called to at the end of the job  ------------
void TestWithTracks::endJob() {
  cout << " End PixelTracksTest, events =  " << countEvents << endl;

  if (countEvents > 0.) {
    countTracks /= countEvents;
    countGoodTracks /= countEvents;
    countTracksInPix /= countEvents;
    countPVs /= countEvents;
    countLumi /= 1000.;
    cout << " Average tracks/event " << countTracks << " good " << countGoodTracks << " in pix " << countTracksInPix
         << " PVs " << countPVs << " events " << countEvents << " lumi pb-1 " << countLumi << "/10, bug!" << endl;
  }
}
//////////////////////////////////////////////////////////////////
// Functions that gets called by framework every event
void TestWithTracks::analyze(const edm::Event &e, const edm::EventSetup &es) {
  using namespace edm;
  using namespace reco;
  static LuminosityBlockNumber_t lumiBlockOld = -9999;

  const float CLU_SIZE_PT_CUT = 1.;

  int trackNumber = 0;
  int countNiceTracks = 0;
  int countPixTracks = 0;

  int numberOfClusters = 0;
  int numberOfPixels = 0;

  int numOfClusPerTrk1 = 0;
  int numOfClustersPerLay1 = 0;
  int numOfPixelsPerLay1 = 0;

  int numOfClusPerTrk2 = 0;
  int numOfClustersPerLay2 = 0;
  int numOfPixelsPerLay2 = 0;

  int numOfClusPerTrk3 = 0;
  int numOfClustersPerLay3 = 0;
  int numOfPixelsPerLay3 = 0;

  int numOfClusPerTrk4 = 0;
  int numOfClustersPerLay4 = 0;
  int numOfPixelsPerLay4 = 0;

  int numOfClusPerTrk5 = 0;
  int numOfClustersPerLay5 = 0;
  int numOfPixelsPerLay5 = 0;

  int numOfClustersPerDisk1 = 0;
  int numOfClustersPerDisk2 = 0;
  int numOfClustersPerDisk3 = 0;
  int numOfClustersPerDisk4 = 0;

  RunNumber_t const run = e.id().run();
  EventNumber_t const event = e.id().event();
  LuminosityBlockNumber_t const lumiBlock = e.luminosityBlock();

  int bx = e.bunchCrossing();
  //int orbit     = e.orbitNumber(); // unused

  if (PRINT)
    cout << "Run " << run << " Event " << event << " LS " << lumiBlock << endl;

  hbx0->Fill(float(bx));
  hlumi0->Fill(float(lumiBlock));

  edm::LuminosityBlock const &iLumi = e.getLuminosityBlock();
  edm::Handle<LumiSummary> lumi;
  iLumi.getByToken(lumiToken_, lumi);
  edm::Handle<edm::ConditionsInLumiBlock> cond;
  float intlumi = 0, instlumi = 0;
  int beamint1 = 0, beamint2 = 0;
  iLumi.getByToken(condToken_, cond);
  // This will only work when running on RECO until (if) they fix it in the FW
  // When running on RAW and reconstructing, the LumiSummary will not appear
  // in the event before reaching endLuminosityBlock(). Therefore, it is not
  // possible to get this info in the event
  if (lumi.isValid()) {
    intlumi = (lumi->intgRecLumi()) / 1000.;  // 10^30 -> 10^33/cm2/sec  ->  1/nb/sec
    instlumi = (lumi->avgInsDelLumi()) / 1000.;
    beamint1 = (cond->totalIntensityBeam1) / 1000;
    beamint2 = (cond->totalIntensityBeam2) / 1000;
  } else {
    //std::cout << "** ERROR: Event does not get lumi info\n";
  }

  hinstl->Fill(float(lumiBlock), float(instlumi));
  hintgl->Fill(float(lumiBlock), float(intlumi));
  hbeam1->Fill(float(lumiBlock), float(beamint1));
  hbeam2->Fill(float(lumiBlock), float(beamint2));

#ifdef L1
  // Get L1
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  e.getByToken(l1gtrrToken_, L1GTRR);

  if (L1GTRR.isValid()) {
    //bool l1a = L1GTRR->decision();  // global decission?
    //cout<<" L1 status :"<<l1a<<" "<<hex;
    for (unsigned int i = 0; i < L1GTRR->decisionWord().size(); ++i) {
      int l1flag = L1GTRR->decisionWord()[i];
      int t1flag = L1GTRR->technicalTriggerWord()[i];

      if (l1flag > 0)
        hl1a->Fill(float(i));
      if (t1flag > 0 && i < 64)
        hl1t->Fill(float(i));
    }  // for loop
  }    // if l1a
#endif

#ifdef HLT

  bool hlt[256];
  for (int i = 0; i < 256; ++i)
    hlt[i] = false;

  edm::TriggerNames TrigNames;
  edm::Handle<edm::TriggerResults> HLTResults;

  // Extract the HLT results
  e.getByToken(hltToken_, HLTResults);
  if ((HLTResults.isValid() == true) && (HLTResults->size() > 0)) {
    //TrigNames.init(*HLTResults);
    const edm::TriggerNames &TrigNames = e.triggerNames(*HLTResults);

    //cout<<TrigNames.triggerNames().size()<<endl;

    for (unsigned int i = 0; i < TrigNames.triggerNames().size(); i++) {  // loop over trigger
      //if(countAllEvents==1) cout<<i<<" "<<TrigNames.triggerName(i)<<endl;

      if ((HLTResults->wasrun(TrigNames.triggerIndex(TrigNames.triggerName(i))) == true) &&
          (HLTResults->accept(TrigNames.triggerIndex(TrigNames.triggerName(i))) == true) &&
          (HLTResults->error(TrigNames.triggerIndex(TrigNames.triggerName(i))) == false)) {
        hlt[i] = true;
        hlt1->Fill(float(i));

      }  // if hlt

    }  // loop
  }    // if valid
#endif

  // Get event setup
  edm::ESHandle<TrackerGeometry> geom = es.getHandle(trackerGeomToken_);
  const TrackerGeometry &theTracker(*geom);

  // -- Primary vertices
  // ----------------------------------------------------------------------
  edm::Handle<reco::VertexCollection> vertices;
  e.getByToken(vtxToken_, vertices);

  if (PRINT)
    cout << " PV list " << vertices->size() << endl;
  int pvNotFake = 0, pvsTrue = 0;
  vector<float> pvzVector;
  for (reco::VertexCollection::const_iterator iv = vertices->begin(); iv != vertices->end(); ++iv) {
    if ((iv->isFake()) == 1)
      continue;
    pvNotFake++;
    float pvx = iv->x();
    float pvy = iv->y();
    float pvz = iv->z();
    int numTracksPerPV = iv->tracksSize();
    //int numTracksPerPV = iv->nTracks();

    //float xe = iv->xError();
    //float ye = iv->yError();
    //float ze = iv->zError();
    //int chi2 = iv->chi2();
    //int dof = iv->ndof();

    if (PRINT)
      cout << " PV " << pvNotFake << " pos = " << pvx << "/" << pvy << "/" << pvz << ", Num of tracks "
           << numTracksPerPV << endl;

    hpvz->Fill(pvz);
    if (pvz > -22. && pvz < 22.) {
      float pvr = sqrt(pvx * pvx + pvy * pvy);
      hpvxy->Fill(pvx, pvy);
      hpvr->Fill(pvr);
      if (pvr < 0.3) {
        pvsTrue++;
        pvzVector.push_back(pvz);
        //if(PRINT) cout<<"PV "<<pvsTrue<<" "<<pvz<<endl;
      }  //pvr
    }    // pvz

    //if(pvsTrue<1) continue; // skip events with no PV

  }  // loop pvs
  hNumPv->Fill(float(pvNotFake));
  hNumPvClean->Fill(float(pvsTrue));

  if (PRINT)
    cout << " Not fake PVs = " << pvNotFake << " good position " << pvsTrue << endl;

  // -- Tracks
  // ----------------------------------------------------------------------
  Handle<reco::TrackCollection> recTracks;
  // e.getByLabel("generalTracks", recTracks);
  // e.getByLabel("ctfWithMaterialTracksP5", recTracks);
  // e.getByLabel("splittedTracksP5", recTracks);
  //e.getByLabel("cosmictrackfinderP5", recTracks);
  e.getByToken(srcToken_, recTracks);

  if (PRINT)
    cout << " Tracks " << recTracks->size() << endl;
  for (reco::TrackCollection::const_iterator t = recTracks->begin(); t != recTracks->end(); ++t) {
    trackNumber++;
    numOfClusPerTrk1 = 0;  // this is confusing, it is used as clus per track
    numOfClusPerTrk2 = 0;
    numOfClusPerTrk3 = 0;
    numOfClusPerTrk4 = 0;
    numOfClusPerTrk5 = 0;
    int pixelHits = 0;

    int size = t->recHitsSize();
    float pt = t->pt();
    float eta = t->eta();
    float phi = t->phi();
    //float trackCharge = t->charge(); // unused
    float d0 = t->d0();
    float dz = t->dz();
    //float tkvx = t->vx();  // unused
    //float tkvy = t->vy();
    //float tkvz = t->vz();

    if (PRINT)
      cout << "Track " << trackNumber << " Pt " << pt << " Eta " << eta << " d0/dz " << d0 << " " << dz << " Hits "
           << size << endl;

    hEta->Fill(eta);
    hDz->Fill(dz);
    if (abs(eta) > 2.8 || abs(dz) > 25.)
      continue;  //  skip

    hD0->Fill(d0);
    if (d0 > 1.0)
      continue;  // skip

    bool goodTrack = false;
    for (vector<float>::iterator m = pvzVector.begin(); m != pvzVector.end(); ++m) {
      float z = *m;
      float tmp = abs(z - dz);
      hzdiff->Fill(tmp);
      if (tmp < 1.)
        goodTrack = true;
    }

    if (isData && !goodTrack)
      continue;
    countNiceTracks++;
    hPt->Fill(pt);

    // Loop over rechits
    for (trackingRecHit_iterator recHit = t->recHitsBegin(); recHit != t->recHitsEnd(); ++recHit) {
      if (!((*recHit)->isValid()))
        continue;

      if ((*recHit)->geographicalId().det() != DetId::Tracker)
        continue;

      const DetId &hit_detId = (*recHit)->geographicalId();
      uint IntSubDetID = (hit_detId.subdetId());

      // Select pixel rechits
      if (IntSubDetID == 0)
        continue;  // Select ??

      int layer = 0, ladder = 0, zindex = 0, ladderOn = 0, module = 0, shell = 0;
      unsigned int disk = 0;     //1,2,3
      unsigned int blade = 0;    //1-24
      unsigned int zindexF = 0;  //
      unsigned int side = 0;     //size=1 for -z, 2 for +z
      unsigned int panel = 0;    //panel=1

      if (IntSubDetID == PixelSubdetector::PixelBarrel) {  // bpix

        // Pixel detector
        PXBDetId pdetId = PXBDetId(hit_detId);
        //unsigned int detTypeP=pdetId.det();
        //unsigned int subidP=pdetId.subdetId();
        // Barell layer = 1,2,3
        layer = pdetId.layer();
        // Barrel ladder id 1-20,32,44.
        ladder = pdetId.ladder();
        // Barrel Z-index=1,8
        zindex = pdetId.module();
        if (zindex < 5)
          side = 1;
        else
          side = 2;

        // Convert to online
        PixelBarrelName pbn(pdetId);
        // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
        PixelBarrelName::Shell sh = pbn.shell();  //enum
        //sector = pbn.sectorName();
        ladderOn = pbn.ladderName();
        //layerOn  = pbn.layerName();
        module = pbn.moduleName();  // 1 to 4
        //half  = pbn.isHalfModule();
        shell = int(sh);
        // change the module sign for z<0
        if (shell == 1 || shell == 2)
          module = -module;  // make -1 to -4 for -z
                             // change ladeer sign for Outer )x<0)
        if (shell == 1 || shell == 3)
          ladderOn = -ladderOn;

        if (PRINT)
          cout << "barrel layer/ladder/module: " << layer << "/" << ladder << "/" << zindex << endl;

      } else if (IntSubDetID == PixelSubdetector::PixelEndcap) {  // fpix

        PXFDetId pdetId = PXFDetId(hit_detId);
        disk = pdetId.disk();       //1,2,3
        blade = pdetId.blade();     //1-24
        zindexF = pdetId.module();  //
        side = pdetId.side();       //size=1 for -z, 2 for +z
        panel = pdetId.panel();     //panel=1

        if (PRINT)
          cout << " forward det, disk " << disk << ", blade " << blade << ", module " << zindexF << ", side " << side
               << ", panel " << panel << endl;

      } else {     // nothings
        continue;  // skip
      }

      // Get the geom-detector
      const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit *>(theTracker.idToDet(hit_detId));
      //double detZ = theGeomDet->surface().position().z();  // unused
      //double detR = theGeomDet->surface().position().perp(); // unused
      const PixelTopology *topol = &(theGeomDet->specificTopology());  // pixel topology

      //std::vector<SiStripRecHit2D*> output = getRecHitComponents((*recHit).get());
      //std::vector<SiPixelRecHit*> TrkComparison::getRecHitComponents(const TrackingRecHit* rechit){

      const SiPixelRecHit *hit = dynamic_cast<const SiPixelRecHit *>((*recHit));
      //edm::Ref<edmNew::DetSetVector<SiStripCluster> ,SiStripCluster> cluster = hit->cluster();
      // get the edm::Ref to the cluster

      if (hit) {
        if (pt > 1.) {  // eliminate low pt tracks
          // RecHit (recthits are transient, so not available without track refit)
          double xloc = hit->localPosition().x();  // 1st meas coord
          double yloc = hit->localPosition().y();  // 2nd meas coord or zero
          //double zloc = hit->localPosition().z();// up, always zero
          LocalError lerr = hit->localPositionError();
          float lerr_x = sqrt(lerr.xx()) * 1E4;
          float lerr_y = sqrt(lerr.yy()) * 1E4;

          if (layer == 1) {
            recHitXError1->Fill(lerr_x);
            recHitYError1->Fill(lerr_y);
            hErrorXB->Fill(float(ladder + (110 * (side - 1))), lerr_x);
            hErrorYB->Fill(float(ladder + (110 * (side - 1))), lerr_y);
          } else if (layer == 2) {
            recHitXError2->Fill(lerr_x);
            recHitYError2->Fill(lerr_y);
            hErrorXB->Fill(float(ladder + 25 + (110 * (side - 1))), lerr_x);
            hErrorYB->Fill(float(ladder + 25 + (110 * (side - 1))), lerr_y);

          } else if (layer == 3) {
            recHitXError3->Fill(lerr_x);
            recHitYError3->Fill(lerr_y);
            hErrorXB->Fill(float(ladder + 60 + (110 * (side - 1))), lerr_x);
            hErrorYB->Fill(float(ladder + 60 + (110 * (side - 1))), lerr_y);
          } else if ((disk == 2) && (side == 1)) {
            recHitXError4->Fill(lerr_x);
            recHitYError4->Fill(lerr_y);
            hErrorXF->Fill(float(blade), lerr_x);
            hErrorYF->Fill(float(blade), lerr_y);

          } else if ((disk == 1) && (side == 1)) {
            recHitXError5->Fill(lerr_x);
            recHitYError5->Fill(lerr_y);
            hErrorXF->Fill(float(blade + 25), lerr_x);
            hErrorYF->Fill(float(blade + 25), lerr_y);

          } else if ((disk == 1) && (side == 2)) {
            recHitXError6->Fill(lerr_x);
            recHitYError6->Fill(lerr_y);
            hErrorXF->Fill(float(blade + 50), lerr_x);
            hErrorYF->Fill(float(blade + 50), lerr_y);
          } else if ((disk == 2) && (side == 2)) {
            recHitXError7->Fill(lerr_x);
            recHitYError7->Fill(lerr_y);
            hErrorXF->Fill(float(blade + 75), lerr_x);
            hErrorYF->Fill(float(blade + 75), lerr_y);
          }

          LocalError lape = theGeomDet->localAlignmentError();
          if (lape.valid()) {
            float tmp11 = 0.;
            if (lape.xx() > 0.)
              tmp11 = sqrt(lape.xx()) * 1E4;
            //float tmp12= sqrt(lape.xy())*1E4;
            float tmp13 = 0.;
            if (lape.yy() > 0.)
              tmp13 = sqrt(lape.yy()) * 1E4;
            //bool tmp14=tmp2<tmp1;
            if (layer == 1) {
              recHitXAlignError1->Fill(tmp11);
              recHitYAlignError1->Fill(tmp13);
              hAErrorXB->Fill(float(ladder + (110 * (side - 1))), tmp11);
              hAErrorYB->Fill(float(ladder + (110 * (side - 1))), tmp13);
            } else if (layer == 2) {
              recHitXAlignError2->Fill(tmp11);
              recHitYAlignError2->Fill(tmp13);
              hAErrorXB->Fill(float(ladder + 25 + (110 * (side - 1))), tmp11);
              hAErrorYB->Fill(float(ladder + 25 + (110 * (side - 1))), tmp13);
            } else if (layer == 3) {
              recHitXAlignError3->Fill(tmp11);
              recHitYAlignError3->Fill(tmp13);
              hAErrorXB->Fill(float(ladder + 60 + (110 * (side - 1))), tmp11);
              hAErrorYB->Fill(float(ladder + 60 + (110 * (side - 1))), tmp13);

            } else if ((disk == 2) && (side == 1)) {
              recHitXAlignError4->Fill(tmp11);
              recHitYAlignError4->Fill(tmp13);
              hAErrorXF->Fill(float(blade), tmp11);
              hAErrorYF->Fill(float(blade), tmp13);

            } else if ((disk == 1) && (side == 1)) {
              recHitXAlignError5->Fill(tmp11);
              recHitYAlignError5->Fill(tmp13);
              hAErrorXF->Fill(float(blade + 25), tmp11);
              hAErrorYF->Fill(float(blade + 25), tmp13);
            } else if ((disk == 1) && (side == 2)) {
              recHitXAlignError6->Fill(tmp11);
              recHitYAlignError6->Fill(tmp13);
              hAErrorXF->Fill(float(blade + 50), tmp11);
              hAErrorYF->Fill(float(blade + 50), tmp13);
            } else if ((disk == 2) && (side == 2)) {
              recHitXAlignError7->Fill(tmp11);
              recHitYAlignError7->Fill(tmp13);
              hAErrorXF->Fill(float(blade + 75), tmp11);
              hAErrorYF->Fill(float(blade + 75), tmp13);
            }

            //cout<<tTopo->pxbLayer(detId)<<" "<<tTopo->pxbModule(detId)<<" "<<rows<<" "<<tmp14<<" "
            if (PRINT)
              cout << " align error " << layer << tmp11 << " " << tmp13 << endl;
          } else {
            cout << " lape = 0" << endl;
          }  // if lape

          if (PRINT)
            cout << " rechit loc " << xloc << " " << yloc << " " << lerr_x << " " << lerr_y << endl;
        }  // limit pt

        edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const &clust = hit->cluster();
        //  check if the ref is not null
        if (!clust.isNonnull())
          continue;

        numberOfClusters++;
        pixelHits++;
        float charge = (clust->charge()) / 1000.0;  // convert electrons to kilo-electrons
        int size = clust->size();
        int sizeX = clust->sizeX();
        int sizeY = clust->sizeY();
        float row = clust->x();
        float col = clust->y();
        numberOfPixels += size;

        //cout<<" clus loc "<<row<<" "<<col<<endl;

        if (PRINT)
          cout << " cluster " << numberOfClusters << " charge = " << charge << " size = " << size << endl;

        LocalPoint lp = topol->localPosition(MeasurementPoint(clust->x(), clust->y()));
        //float x = lp.x();
        //float y = lp.y();
        //cout<<" clu loc "<<x<<" "<<y<<endl;

        GlobalPoint clustgp = theGeomDet->surface().toGlobal(lp);
        double gX = clustgp.x();
        double gY = clustgp.y();
        double gZ = clustgp.z();

        //cout<<" CLU GLOBAL "<<gX<<" "<<gY<<" "<<gZ<<endl;

        TVector3 v(gX, gY, gZ);
        //float phi = v.Phi(); // unused

        //int maxPixelCol = clust->maxPixelCol();
        //int maxPixelRow = clust->maxPixelRow();
        //int minPixelCol = clust->minPixelCol();
        //int minPixelRow = clust->minPixelRow();
        //int geoId = PixGeom->geographicalId().rawId();
        // Replace with the topology methods
        // edge method moved to topologi class
        //int edgeHitX = (int) ( topol->isItEdgePixelInX( minPixelRow ) || topol->isItEdgePixelInX( maxPixelRow ) );
        //int edgeHitY = (int) ( topol->isItEdgePixelInY( minPixelCol ) || topol->isItEdgePixelInY( maxPixelCol ) );

        // calculate alpha and beta from cluster position
        //LocalTrajectoryParameters ltp = tsos.localParameters();
        //LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
        //float locx = localDir.x();
        //float locy = localDir.y();
        //float locz = localDir.z();
        //float loctheta = localDir.theta(); // currently unused
        //float alpha = atan2( locz, locx );
        //float beta = atan2( locz, locy );

        if (layer == 1) {
          hDetMap1->Fill(float(zindex), float(ladder));
          hcluDetMap1->Fill(col, row);
          hcharge1->Fill(charge);
          //hcols1->Fill(col);
          //hrows1->Fill(row);

          hclusMap1->Fill(gZ, phi);
          hmult1->Fill(zindex, float(size));

          if (pt > CLU_SIZE_PT_CUT) {
            hsize1->Fill(float(size));
            hsizex1->Fill(float(sizeX));
            hsizey1->Fill(float(sizeY));

            hclumult1->Fill(eta, float(size));
            hclumultx1->Fill(eta, float(sizeX));
            hclumulty1->Fill(eta, float(sizeY));
            hcluchar1->Fill(eta, float(charge));

            //cout<<ladder<<" "<<ladderOn<<endl;

            hclumultld1->Fill(float(ladderOn), size);
            hclumultxld1->Fill(float(ladderOn), sizeX);
            hclumultyld1->Fill(float(ladderOn), sizeY);
            hclucharld1->Fill(float(ladderOn), charge);
          }

#ifdef VDM_STUDIES
          hcharCluls->Fill(lumiBlock, charge);
          hsizeCluls->Fill(lumiBlock, size);
          hsizeXCluls->Fill(lumiBlock, sizeX);
          hcharCluls1->Fill(lumiBlock, charge);
          hsizeCluls1->Fill(lumiBlock, size);
          hsizeXCluls1->Fill(lumiBlock, sizeX);
#endif

          numOfClusPerTrk1++;
          numOfClustersPerLay1++;
          numOfPixelsPerLay1 += size;

        } else if (layer == 2) {
          hDetMap2->Fill(float(zindex), float(ladder));
          hcluDetMap2->Fill(col, row);
          hcharge2->Fill(charge);
          //hcols2->Fill(col);
          //hrows2->Fill(row);

          hclusMap2->Fill(gZ, phi);
          hmult2->Fill(zindex, float(size));

          if (pt > CLU_SIZE_PT_CUT) {
            hsize2->Fill(float(size));
            hsizex2->Fill(float(sizeX));
            hsizey2->Fill(float(sizeY));

            hclumult2->Fill(eta, float(size));
            hclumultx2->Fill(eta, float(sizeX));
            hclumulty2->Fill(eta, float(sizeY));
            hcluchar2->Fill(eta, float(charge));

            hclumultld2->Fill(float(ladderOn), size);
            hclumultxld2->Fill(float(ladderOn), sizeX);
            hclumultyld2->Fill(float(ladderOn), sizeY);
            hclucharld2->Fill(float(ladderOn), charge);
          }

#ifdef VDM_STUDIES
          hcharCluls->Fill(lumiBlock, charge);
          hsizeCluls->Fill(lumiBlock, size);
          hsizeXCluls->Fill(lumiBlock, sizeX);
          hcharCluls2->Fill(lumiBlock, charge);
          hsizeCluls2->Fill(lumiBlock, size);
          hsizeXCluls2->Fill(lumiBlock, sizeX);
#endif

          numOfClusPerTrk2++;
          numOfClustersPerLay2++;
          numOfPixelsPerLay2 += size;

        } else if (layer == 3) {
          hDetMap3->Fill(float(zindex), float(ladder));
          hcluDetMap3->Fill(col, row);
          hcharge3->Fill(charge);
          //hcols3->Fill(col);
          //hrows3->Fill(row);

          hclusMap3->Fill(gZ, phi);
          hmult3->Fill(zindex, float(size));

          if (pt > CLU_SIZE_PT_CUT) {
            hsize3->Fill(float(size));
            hsizex3->Fill(float(sizeX));
            hsizey3->Fill(float(sizeY));
            hclumult3->Fill(eta, float(size));
            hclumultx3->Fill(eta, float(sizeX));
            hclumulty3->Fill(eta, float(sizeY));
            hcluchar3->Fill(eta, float(charge));

            hclumultld3->Fill(float(ladderOn), size);
            hclumultxld3->Fill(float(ladderOn), sizeX);
            hclumultyld3->Fill(float(ladderOn), sizeY);
            hclucharld3->Fill(float(ladderOn), charge);
          }

#ifdef VDM_STUDIES
          hcharCluls->Fill(lumiBlock, charge);
          hsizeCluls->Fill(lumiBlock, size);
          hsizeXCluls->Fill(lumiBlock, sizeX);
          hcharCluls3->Fill(lumiBlock, charge);
          hsizeCluls3->Fill(lumiBlock, size);
          hsizeXCluls3->Fill(lumiBlock, sizeX);
#endif

          numOfClusPerTrk3++;
          numOfClustersPerLay3++;
          numOfPixelsPerLay3 += size;

        } else if (disk == 1) {
          numOfClusPerTrk4++;
          numOfClustersPerLay4++;
          numOfPixelsPerLay4 += size;

          hcharge4->Fill(charge);
          if (pt > CLU_SIZE_PT_CUT) {
            hsize4->Fill(float(size));
            hsizex4->Fill(float(sizeX));
            hsizey4->Fill(float(sizeY));
          }

          if (side == 1)
            numOfClustersPerDisk2++;  // -z
          else if (side == 2)
            numOfClustersPerDisk3++;  // +z

        } else if (disk == 2) {
          numOfClusPerTrk5++;
          numOfClustersPerLay5++;
          numOfPixelsPerLay5 += size;

          hcharge5->Fill(charge);
          if (pt > CLU_SIZE_PT_CUT) {
            hsize5->Fill(float(size));
            hsizex5->Fill(float(sizeX));
            hsizey5->Fill(float(sizeY));
          }

          if (side == 1)
            numOfClustersPerDisk1++;  // -z
          else if (side == 2)
            numOfClustersPerDisk4++;  // +z

        } else {
          cout << " which layer is this? " << layer << " " << disk << endl;
        }  // if layer

      }  // if valid

    }  // clusters

    if (pixelHits > 0)
      countPixTracks++;

    if (PRINT)
      cout << " Clusters for track " << trackNumber << " num of clusters " << numberOfClusters << " num of pixels "
           << pixelHits << endl;

#ifdef HISTOS
    // per track histos
    if (numberOfClusters > 0) {
      hclusPerTrk1->Fill(float(numOfClusPerTrk1));
      if (PRINT)
        cout << "Lay1: number of clusters per track = " << numOfClusPerTrk1 << endl;
      hclusPerTrk2->Fill(float(numOfClusPerTrk2));
      if (PRINT)
        cout << "Lay2: number of clusters per track = " << numOfClusPerTrk1 << endl;
      hclusPerTrk3->Fill(float(numOfClusPerTrk3));
      if (PRINT)
        cout << "Lay3: number of clusters per track = " << numOfClusPerTrk1 << endl;
      hclusPerTrk4->Fill(float(numOfClusPerTrk4));  // fpix  disk1
      hclusPerTrk5->Fill(float(numOfClusPerTrk5));  // fpix disk2

      float clusPerTrkB = numOfClusPerTrk1 + numOfClusPerTrk2 + numOfClusPerTrk3;
      float clusPerTrkF = numOfClusPerTrk4 + numOfClusPerTrk5;
      float clusPerTrk = clusPerTrkB + clusPerTrkF;

      hclusPerTrkB->Fill(clusPerTrkB);
      hclusPerTrkF->Fill(clusPerTrkF);
      hclusPerTrk->Fill(clusPerTrk);

      hclusPerTrkVsEta->Fill(eta, clusPerTrk);
      hclusPerTrkVsEtaB->Fill(eta, clusPerTrkB);
      hclusPerTrkVsEtaF->Fill(eta, clusPerTrkF);
      hclusPerTrkVsPt->Fill(pt, clusPerTrk);
      hclusPerTrkVsls->Fill(lumiBlock, clusPerTrk);
    }
#endif  // HISTOS

  }  // tracks

#ifdef HISTOS
  // total layer histos
  if (numberOfClusters > 0) {
    hclusPerLay1->Fill(float(numOfClustersPerLay1));
    hclusPerLay2->Fill(float(numOfClustersPerLay2));
    hclusPerLay3->Fill(float(numOfClustersPerLay3));

    hclusPerDisk1->Fill(float(numOfClustersPerDisk1));
    hclusPerDisk2->Fill(float(numOfClustersPerDisk2));
    hclusPerDisk3->Fill(float(numOfClustersPerDisk3));
    hclusPerDisk4->Fill(float(numOfClustersPerDisk4));

    //hdetsPerLay1->Fill(float(numberOfDetUnits1));
    //hdetsPerLay2->Fill(float(numberOfDetUnits2));
    //hdetsPerLay3->Fill(float(numberOfDetUnits3));
    //int tmp = numberOfDetUnits1+numberOfDetUnits2+numberOfDetUnits3;
    //hpixPerLay1->Fill(float(numOfPixelsPerLay1));
    //hpixPerLay2->Fill(float(numOfPixelsPerLay2));
    //hpixPerLay3->Fill(float(numOfPixelsPerLay3));
    //htest7->Fill(float(tmp));
    hclusBpix->Fill(float(numberOfClusters));
    hpixBpix->Fill(float(numberOfPixels));
  }
  htracksGood->Fill(float(countNiceTracks));
  htracksGoodInPix->Fill(float(countPixTracks));
  htracks->Fill(float(trackNumber));

  hbx->Fill(float(bx));
  hlumi->Fill(float(lumiBlock));

  htracksls->Fill(float(lumiBlock), float(countPixTracks));
  hpvsls->Fill(float(lumiBlock), float(pvsTrue));
  if (instlumi > 0.) {
    float tmp = float(countPixTracks) / instlumi;
    htrackslsn->Fill(float(lumiBlock), tmp);
    tmp = float(pvsTrue) / instlumi;
    hpvslsn->Fill(float(lumiBlock), tmp);
  }

#ifdef VDM_STUDIES

  hclusls->Fill(float(lumiBlock), float(numberOfClusters));  // clusters fpix+bpix
  //hpixls->Fill(float(lumiBlock),float(numberOfPixels)); // pixels fpix+bpix

  hclubx->Fill(float(bx), float(numberOfClusters));  // clusters fpix+bpix
  //hpixbx->Fill(float(bx),float(numberOfPixels)); // pixels fpix+bpix
  hpvbx->Fill(float(bx), float(pvsTrue));            // pvs
  htrackbx->Fill(float(bx), float(countPixTracks));  // tracks

  hclusls1->Fill(float(lumiBlock), float(numOfClustersPerLay1));  // clusters bpix1
  //hpixls1->Fill( float(lumiBlock),float(numOfPixPerLay1)); // pixels bpix1
  hclusls2->Fill(float(lumiBlock), float(numOfClustersPerLay2));  // clusters bpix2
  //hpixls2->Fill( float(lumiBlock),float(numOfPixPerLay2)); // pixels bpix2
  hclusls3->Fill(float(lumiBlock), float(numOfClustersPerLay3));  // clusters bpix3
  //hpixls3->Fill( float(lumiBlock),float(numOfPixPerLay3)); // pixels bpix3
#endif

#endif  // HISTOS

  //
  countTracks += float(trackNumber);
  countGoodTracks += float(countNiceTracks);
  countTracksInPix += float(countPixTracks);
  countPVs += float(pvsTrue);
  countEvents++;
  if (lumiBlock != lumiBlockOld) {
    countLumi += intlumi;
    lumiBlockOld = lumiBlock;
  }

  if (PRINT)
    cout << " event with tracks = " << trackNumber << " " << countNiceTracks << endl;

  return;

#ifdef USE_TRAJ
  // Not used

  //----------------------------------------------------------------------------
  // Use Trajectories

  edm::Handle<TrajTrackAssociationCollection> trajTrackCollectionHandle;
  e.getByToken(trackAssocToken_, trajTrackCollectionHandle);

  TrajectoryStateCombiner tsoscomb;

  int NbrTracks = trajTrackCollectionHandle->size();
  std::cout << " track measurements " << trajTrackCollectionHandle->size() << std::endl;

  int trackNumber = 0;
  int numberOfClusters = 0;

  for (TrajTrackAssociationCollection::const_iterator it = trajTrackCollectionHandle->begin(),
                                                      itEnd = trajTrackCollectionHandle->end();
       it != itEnd;
       ++it) {
    int pixelHits = 0;
    int stripHits = 0;
    const Track &track = *it->val;
    const Trajectory &traj = *it->key;

    std::vector<TrajectoryMeasurement> checkColl = traj.measurements();
    for (std::vector<TrajectoryMeasurement>::const_iterator checkTraj = checkColl.begin(); checkTraj != checkColl.end();
         ++checkTraj) {
      if (!checkTraj->updatedState().isValid())
        continue;
      TransientTrackingRecHit::ConstRecHitPointer testhit = checkTraj->recHit();
      if (!testhit->isValid() || testhit->geographicalId().det() != DetId::Tracker)
        continue;
      uint testSubDetID = (testhit->geographicalId().subdetId());
      if (testSubDetID == PixelSubdetector::PixelBarrel || testSubDetID == PixelSubdetector::PixelEndcap)
        pixelHits++;
      else if (testSubDetID == StripSubdetector::TIB || testSubDetID == StripSubdetector::TOB ||
               testSubDetID == StripSubdetector::TID || testSubDetID == StripSubdetector::TEC)
        stripHits++;
    }

    if (pixelHits == 0)
      continue;

    trackNumber++;
    std::cout << " track " << trackNumber << " has pixelhits " << pixelHits << std::endl;
    pixelHits = 0;

    //std::vector<TrajectoryMeasurement> tmColl = traj.measurements();
    for (std::vector<TrajectoryMeasurement>::const_iterator itTraj = checkColl.begin(); itTraj != checkColl.end();
         ++itTraj) {
      if (!itTraj->updatedState().isValid())
        continue;

      TrajectoryStateOnSurface tsos = tsoscomb(itTraj->forwardPredictedState(), itTraj->backwardPredictedState());
      TransientTrackingRecHit::ConstRecHitPointer hit = itTraj->recHit();
      if (!hit->isValid() || hit->geographicalId().det() != DetId::Tracker)
        continue;

      const DetId &hit_detId = hit->geographicalId();
      uint IntSubDetID = (hit_detId.subdetId());

      if (IntSubDetID == 0)
        continue;  // Select ??
      if (IntSubDetID != PixelSubdetector::PixelBarrel)
        continue;  // look only at bpix || IntSubDetID == PixelSubdetector::PixelEndcap) {

      //       const GeomDetUnit* detUnit = hit->detUnit();
      //       if(detUnit) {
      // 	const Surface& surface = hit->detUnit()->surface();
      // 	const TrackerGeometry& theTracker(*tkGeom_);
      // 	const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDet(hit_detId) );
      // 	const RectangularPixelTopology * topol = dynamic_cast<const RectangularPixelTopology*>(&(theGeomDet->specificTopology()));
      //       }

      // get the enclosed persistent hit
      const TrackingRecHit *persistentHit = hit->hit();
      // check if it's not null, and if it's a valid pixel hit
      if ((persistentHit != 0) && (typeid(*persistentHit) == typeid(SiPixelRecHit))) {
        // tell the C++ compiler that the hit is a pixel hit
        const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit *>(hit->hit());
        // get the edm::Ref to the cluster
        edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const &clust = (*pixhit).cluster();
        //  check if the ref is not null
        if (clust.isNonnull()) {
          numberOfClusters++;
          pixelHits++;
          float charge = (clust->charge()) / 1000.0;  // convert electrons to kilo-electrons
          int size = clust->size();
          int size_x = clust->sizeX();
          int size_y = clust->sizeY();
          float row = clust->x();
          float col = clust->y();

          //LocalPoint lp = topol->localPosition(MeasurementPoint(clust_.row,clust_.col));
          //float x = lp.x();
          //float y = lp.y();

          int maxPixelCol = clust->maxPixelCol();
          int maxPixelRow = clust->maxPixelRow();
          int minPixelCol = clust->minPixelCol();
          int minPixelRow = clust->minPixelRow();

          //int geoId = PixGeom->geographicalId().rawId();

          // Replace with the topology methods
          // edge method moved to topologi class
          //int edgeHitX = (int) ( topol->isItEdgePixelInX( minPixelRow ) || topol->isItEdgePixelInX( maxPixelRow ) );
          //int edgeHitY = (int) ( topol->isItEdgePixelInY( minPixelCol ) || topol->isItEdgePixelInY( maxPixelCol ) );

          // calculate alpha and beta from cluster position
          //LocalTrajectoryParameters ltp = tsos.localParameters();
          //LocalVector localDir = ltp.momentum()/ltp.momentum().mag();

          //float locx = localDir.x();
          //float locy = localDir.y();
          //float locz = localDir.z();
          //float loctheta = localDir.theta(); // currently unused

          //float alpha = atan2( locz, locx );
          //float beta = atan2( locz, locy );

          //clust_.normalized_charge = clust_.charge*sqrt(1.0/(1.0/pow(tan(clust_.clust_alpha),2)+1.0/pow(tan(clust_.clust_beta),2)+1.0));
        }  // valid cluster
      }    // valid peristant hit

    }  // loop over trajectory meas.

    if (PRINT)
      cout << " Cluster for track " << trackNumber << " cluaters " << numberOfClusters << " " << pixelHits << endl;

  }  // loop over tracks

#endif  // USE_TRAJ

  cout << " event with tracks = " << trackNumber << " " << countGoodTracks << endl;

}  // end

//define this as a plug-in
DEFINE_FWK_MODULE(TestWithTracks);
