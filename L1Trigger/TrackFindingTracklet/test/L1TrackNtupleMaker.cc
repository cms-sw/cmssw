//////////////////////////////////////////////////////////////////////
//                                                                  //
//  Analyzer for making mini-ntuple for L1 track performance plots  //
//                                                                  //
//////////////////////////////////////////////////////////////////////

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

////////////////
// PHYSICS TOOLS
#include "L1Trigger/TrackFindingTracklet/interface/HitPatternHelper.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CLHEP/Units/PhysicalConstants.h"

///////////////
// ROOT HEADERS
#include <TROOT.h>
#include <TCanvas.h>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>

//////////////
// NAMESPACES
using namespace std;
using namespace edm;

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1TrackNtupleMaker : public one::EDAnalyzer<one::WatchRuns, one::SharedResources> {
public:
  // Constructor/destructor
  explicit L1TrackNtupleMaker(const edm::ParameterSet& iConfig);
  ~L1TrackNtupleMaker() override;

  // Mandatory methods
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void beginRun(const Run& iEvent, const EventSetup& iSetup) override {}
  void endRun(const Run& iEvent, const EventSetup& iSetup) override {}

protected:
private:
  //-----------------------------------------------------------------------------------------------
  // Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  int MyProcess;       // 11/13/211 for single electrons/muons/pions, 6/15 for pions from ttbar/taus, 1 for inclusive
  bool DebugMode;      // lots of debug printout statements
  bool SaveAllTracks;  // store in ntuples not only truth-matched tracks but ALL tracks
  bool SaveStubs;      // option to save also stubs in the ntuples (makes them large...)
  int L1Tk_nPar;       // use 4 or 5 parameter track fit?
  int TP_minNStub;  // require TPs to have >= minNStub (defining efficiency denominator) (==0 means to only require >= 1 cluster)
  int TP_minNStubLayer;  // require TPs to have stubs in >= minNStubLayer layers/disks (defining efficiency denominator)
  double TP_minPt;       // save TPs with pt > minPt
  double TP_maxEta;      // save TPs with |eta| < maxEta
  double TP_maxZ0;       // save TPs with |z0| < maxZ0
  int L1Tk_minNStub;     // require L1 tracks to have >= minNStub (this is mostly for tracklet purposes)

  bool TrackingInJets;  // do tracking in jets?

  edm::InputTag L1TrackInputTag;       // L1 track collection
  edm::InputTag MCTruthTrackInputTag;  // MC truth collection
  edm::InputTag MCTruthClusterInputTag;
  edm::InputTag L1StubInputTag;
  edm::InputTag MCTruthStubInputTag;
  edm::InputTag TrackingParticleInputTag;
  edm::InputTag TrackingVertexInputTag;
  edm::InputTag GenJetInputTag;

  edm::EDGetTokenT<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_> > > ttClusterToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> > > ttStubToken_;
  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_> > ttClusterMCTruthToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_> > ttStubMCTruthToken_;

  edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > ttTrackToken_;
  edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > ttTrackMCTruthToken_;

  edm::EDGetTokenT<std::vector<TrackingParticle> > TrackingParticleToken_;
  edm::EDGetTokenT<std::vector<TrackingVertex> > TrackingVertexToken_;

  edm::EDGetTokenT<std::vector<reco::GenJet> > GenJetToken_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeom_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopo_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> getTokenBField_;
  edm::ESGetToken<hph::Setup, hph::SetupRcd> getTokenHPHSetup_;
  //-----------------------------------------------------------------------------------------------
  // tree & branches for mini-ntuple

  bool available_;  // ROOT file for histograms is open.

  TTree* eventTree;

  // all L1 tracks
  std::vector<float>* m_trk_pt;
  std::vector<float>* m_trk_eta;
  std::vector<float>* m_trk_phi;
  std::vector<float>* m_trk_d0;  // (filled if L1Tk_nPar==5, else 999)
  std::vector<float>* m_trk_z0;
  std::vector<float>* m_trk_chi2;
  std::vector<float>* m_trk_chi2rphi;
  std::vector<float>* m_trk_chi2rz;
  std::vector<float>* m_trk_bendchi2;
  std::vector<int>* m_trk_nstub;
  std::vector<int>* m_trk_lhits;
  std::vector<int>* m_trk_dhits;
  std::vector<int>* m_trk_seed;
  std::vector<int>* m_trk_hitpattern;
  std::vector<int>* m_trk_lhits_hitpattern;  // 6-digit hit mask (barrel layer only) dervied from hitpattern
  std::vector<int>* m_trk_dhits_hitpattern;  // disk only
  std::vector<int>* m_trk_nPSstub_hitpattern;
  std::vector<int>* m_trk_n2Sstub_hitpattern;
  std::vector<int>* m_trk_nLostPSstub_hitpattern;
  std::vector<int>* m_trk_nLost2Sstub_hitpattern;
  std::vector<int>* m_trk_nLoststub_V1_hitpattern;  // Same as the definiton of "nlaymiss_interior" in TrackQuality.cc
  std::vector<int>* m_trk_nLoststub_V2_hitpattern;  // A tighter version of "nlaymiss_interior"
  std::vector<unsigned int>* m_trk_phiSector;
  std::vector<int>* m_trk_etaSector;
  std::vector<int>* m_trk_genuine;
  std::vector<int>* m_trk_loose;
  std::vector<int>* m_trk_unknown;
  std::vector<int>* m_trk_combinatoric;
  std::vector<int>* m_trk_fake;  //0 fake, 1 track from primary interaction, 2 secondary track
  std::vector<float>* m_trk_MVA1;
  std::vector<int>* m_trk_matchtp_pdgid;
  std::vector<float>* m_trk_matchtp_pt;
  std::vector<float>* m_trk_matchtp_eta;
  std::vector<float>* m_trk_matchtp_phi;
  std::vector<float>* m_trk_matchtp_z0;
  std::vector<float>* m_trk_matchtp_dxy;
  std::vector<float>* m_trk_matchtp_d0;
  std::vector<int>* m_trk_injet;          //is the track within dR<0.4 of a genjet with pt > 30 GeV?
  std::vector<int>* m_trk_injet_highpt;   //is the track within dR<0.4 of a genjet with pt > 100 GeV?
  std::vector<int>* m_trk_injet_vhighpt;  //is the track within dR<0.4 of a genjet with pt > 200 GeV?

  // all tracking particles
  std::vector<float>* m_tp_pt;
  std::vector<float>* m_tp_eta;
  std::vector<float>* m_tp_phi;
  std::vector<float>* m_tp_dxy;
  std::vector<float>* m_tp_d0;
  std::vector<float>* m_tp_z0;
  std::vector<float>* m_tp_d0_prod;
  std::vector<float>* m_tp_z0_prod;
  std::vector<int>* m_tp_pdgid;
  std::vector<int>* m_tp_nmatch;
  std::vector<int>* m_tp_nstub;
  std::vector<int>* m_tp_eventid;
  std::vector<int>* m_tp_charge;
  std::vector<int>* m_tp_injet;
  std::vector<int>* m_tp_injet_highpt;
  std::vector<int>* m_tp_injet_vhighpt;

  // *L1 track* properties if m_tp_nmatch > 0
  std::vector<float>* m_matchtrk_pt;
  std::vector<float>* m_matchtrk_eta;
  std::vector<float>* m_matchtrk_phi;
  std::vector<float>* m_matchtrk_d0;  //this variable is only filled if L1Tk_nPar==5
  std::vector<float>* m_matchtrk_z0;
  std::vector<float>* m_matchtrk_chi2;
  std::vector<float>* m_matchtrk_chi2rphi;
  std::vector<float>* m_matchtrk_chi2rz;
  std::vector<float>* m_matchtrk_bendchi2;
  std::vector<float>* m_matchtrk_MVA1;
  std::vector<int>* m_matchtrk_nstub;
  std::vector<int>* m_matchtrk_lhits;
  std::vector<int>* m_matchtrk_dhits;
  std::vector<int>* m_matchtrk_seed;
  std::vector<int>* m_matchtrk_hitpattern;
  std::vector<int>* m_matchtrk_injet;
  std::vector<int>* m_matchtrk_injet_highpt;
  std::vector<int>* m_matchtrk_injet_vhighpt;

  // ALL stubs
  std::vector<float>* m_allstub_x;
  std::vector<float>* m_allstub_y;
  std::vector<float>* m_allstub_z;

  std::vector<int>* m_allstub_isBarrel;  // stub is in barrel (1) or in disk (0)
  std::vector<int>* m_allstub_layer;
  std::vector<int>* m_allstub_isPSmodule;

  std::vector<float>* m_allstub_trigDisplace;
  std::vector<float>* m_allstub_trigOffset;
  std::vector<float>* m_allstub_trigPos;
  std::vector<float>* m_allstub_trigBend;

  // stub associated with tracking particle ?
  std::vector<int>* m_allstub_matchTP_pdgid;  // -999 if not matched
  std::vector<float>* m_allstub_matchTP_pt;   // -999 if not matched
  std::vector<float>* m_allstub_matchTP_eta;  // -999 if not matched
  std::vector<float>* m_allstub_matchTP_phi;  // -999 if not matched

  std::vector<int>* m_allstub_genuine;

  // track jet variables (for each gen jet, store the sum of pt of TPs / tracks inside jet cone)
  std::vector<float>* m_jet_eta;
  std::vector<float>* m_jet_phi;
  std::vector<float>* m_jet_pt;
  std::vector<float>* m_jet_tp_sumpt;
  std::vector<float>* m_jet_trk_sumpt;
  std::vector<float>* m_jet_matchtrk_sumpt;
};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
L1TrackNtupleMaker::L1TrackNtupleMaker(edm::ParameterSet const& iConfig) : config(iConfig) {
  usesResource("TFileService");
  MyProcess = iConfig.getParameter<int>("MyProcess");
  DebugMode = iConfig.getParameter<bool>("DebugMode");
  SaveAllTracks = iConfig.getParameter<bool>("SaveAllTracks");
  SaveStubs = iConfig.getParameter<bool>("SaveStubs");
  L1Tk_nPar = iConfig.getParameter<int>("L1Tk_nPar");
  TP_minNStub = iConfig.getParameter<int>("TP_minNStub");
  TP_minNStubLayer = iConfig.getParameter<int>("TP_minNStubLayer");
  TP_minPt = iConfig.getParameter<double>("TP_minPt");
  TP_maxEta = iConfig.getParameter<double>("TP_maxEta");
  TP_maxZ0 = iConfig.getParameter<double>("TP_maxZ0");
  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
  MCTruthTrackInputTag = iConfig.getParameter<edm::InputTag>("MCTruthTrackInputTag");
  L1Tk_minNStub = iConfig.getParameter<int>("L1Tk_minNStub");

  TrackingInJets = iConfig.getParameter<bool>("TrackingInJets");

  L1StubInputTag = iConfig.getParameter<edm::InputTag>("L1StubInputTag");
  MCTruthClusterInputTag = iConfig.getParameter<edm::InputTag>("MCTruthClusterInputTag");
  MCTruthStubInputTag = iConfig.getParameter<edm::InputTag>("MCTruthStubInputTag");
  TrackingParticleInputTag = iConfig.getParameter<edm::InputTag>("TrackingParticleInputTag");
  TrackingVertexInputTag = iConfig.getParameter<edm::InputTag>("TrackingVertexInputTag");
  GenJetInputTag = iConfig.getParameter<edm::InputTag>("GenJetInputTag");

  ttTrackToken_ = consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(L1TrackInputTag);
  ttTrackMCTruthToken_ = consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> >(MCTruthTrackInputTag);
  ttStubToken_ = consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> > >(L1StubInputTag);
  ttClusterMCTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_> >(MCTruthClusterInputTag);
  ttStubMCTruthToken_ = consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_> >(MCTruthStubInputTag);

  TrackingParticleToken_ = consumes<std::vector<TrackingParticle> >(TrackingParticleInputTag);
  TrackingVertexToken_ = consumes<std::vector<TrackingVertex> >(TrackingVertexInputTag);
  GenJetToken_ = consumes<std::vector<reco::GenJet> >(GenJetInputTag);

  getTokenTrackerGeom_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  getTokenTrackerTopo_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  getTokenBField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  getTokenHPHSetup_ = esConsumes<hph::Setup, hph::SetupRcd>();
}

/////////////
// DESTRUCTOR
L1TrackNtupleMaker::~L1TrackNtupleMaker() {}

//////////
// END JOB
void L1TrackNtupleMaker::endJob() {
  // things to be done at the exit of the event Loop
  edm::LogVerbatim("Tracklet") << "L1TrackNtupleMaker::endJob";
}

////////////
// BEGIN JOB
void L1TrackNtupleMaker::beginJob() {
  // things to be done before entering the event Loop
  edm::LogVerbatim("Tracklet") << "L1TrackNtupleMaker::beginJob";

  //-----------------------------------------------------------------------------------------------
  // book histograms / make ntuple
  edm::Service<TFileService> fs;
  available_ = fs.isAvailable();
  if (not available_)
    return;  // No ROOT file open.

  // initilize
  m_trk_pt = new std::vector<float>;
  m_trk_eta = new std::vector<float>;
  m_trk_phi = new std::vector<float>;
  m_trk_z0 = new std::vector<float>;
  m_trk_d0 = new std::vector<float>;
  m_trk_chi2 = new std::vector<float>;
  m_trk_chi2rphi = new std::vector<float>;
  m_trk_chi2rz = new std::vector<float>;
  m_trk_bendchi2 = new std::vector<float>;
  m_trk_nstub = new std::vector<int>;
  m_trk_lhits = new std::vector<int>;
  m_trk_dhits = new std::vector<int>;
  m_trk_seed = new std::vector<int>;
  m_trk_hitpattern = new std::vector<int>;
  m_trk_lhits_hitpattern = new std::vector<int>;
  m_trk_dhits_hitpattern = new std::vector<int>;
  m_trk_nPSstub_hitpattern = new std::vector<int>;
  m_trk_n2Sstub_hitpattern = new std::vector<int>;
  m_trk_nLostPSstub_hitpattern = new std::vector<int>;
  m_trk_nLost2Sstub_hitpattern = new std::vector<int>;
  m_trk_nLoststub_V1_hitpattern = new std::vector<int>;
  m_trk_nLoststub_V2_hitpattern = new std::vector<int>;
  m_trk_phiSector = new std::vector<unsigned int>;
  m_trk_etaSector = new std::vector<int>;
  m_trk_genuine = new std::vector<int>;
  m_trk_loose = new std::vector<int>;
  m_trk_unknown = new std::vector<int>;
  m_trk_combinatoric = new std::vector<int>;
  m_trk_fake = new std::vector<int>;
  m_trk_MVA1 = new std::vector<float>;
  m_trk_matchtp_pdgid = new std::vector<int>;
  m_trk_matchtp_pt = new std::vector<float>;
  m_trk_matchtp_eta = new std::vector<float>;
  m_trk_matchtp_phi = new std::vector<float>;
  m_trk_matchtp_z0 = new std::vector<float>;
  m_trk_matchtp_dxy = new std::vector<float>;
  m_trk_matchtp_d0 = new std::vector<float>;
  m_trk_injet = new std::vector<int>;
  m_trk_injet_highpt = new std::vector<int>;
  m_trk_injet_vhighpt = new std::vector<int>;

  m_tp_pt = new std::vector<float>;
  m_tp_eta = new std::vector<float>;
  m_tp_phi = new std::vector<float>;
  m_tp_dxy = new std::vector<float>;
  m_tp_d0 = new std::vector<float>;
  m_tp_z0 = new std::vector<float>;
  m_tp_d0_prod = new std::vector<float>;
  m_tp_z0_prod = new std::vector<float>;
  m_tp_pdgid = new std::vector<int>;
  m_tp_nmatch = new std::vector<int>;
  m_tp_nstub = new std::vector<int>;
  m_tp_eventid = new std::vector<int>;
  m_tp_charge = new std::vector<int>;
  m_tp_injet = new std::vector<int>;
  m_tp_injet_highpt = new std::vector<int>;
  m_tp_injet_vhighpt = new std::vector<int>;

  m_matchtrk_pt = new std::vector<float>;
  m_matchtrk_eta = new std::vector<float>;
  m_matchtrk_phi = new std::vector<float>;
  m_matchtrk_z0 = new std::vector<float>;
  m_matchtrk_d0 = new std::vector<float>;
  m_matchtrk_chi2 = new std::vector<float>;
  m_matchtrk_chi2rphi = new std::vector<float>;
  m_matchtrk_chi2rz = new std::vector<float>;
  m_matchtrk_bendchi2 = new std::vector<float>;
  m_matchtrk_MVA1 = new std::vector<float>;
  m_matchtrk_nstub = new std::vector<int>;
  m_matchtrk_dhits = new std::vector<int>;
  m_matchtrk_lhits = new std::vector<int>;
  m_matchtrk_seed = new std::vector<int>;
  m_matchtrk_hitpattern = new std::vector<int>;
  m_matchtrk_injet = new std::vector<int>;
  m_matchtrk_injet_highpt = new std::vector<int>;
  m_matchtrk_injet_vhighpt = new std::vector<int>;

  m_allstub_x = new std::vector<float>;
  m_allstub_y = new std::vector<float>;
  m_allstub_z = new std::vector<float>;

  m_allstub_isBarrel = new std::vector<int>;
  m_allstub_layer = new std::vector<int>;
  m_allstub_isPSmodule = new std::vector<int>;
  m_allstub_trigDisplace = new std::vector<float>;
  m_allstub_trigOffset = new std::vector<float>;
  m_allstub_trigPos = new std::vector<float>;
  m_allstub_trigBend = new std::vector<float>;

  m_allstub_matchTP_pdgid = new std::vector<int>;
  m_allstub_matchTP_pt = new std::vector<float>;
  m_allstub_matchTP_eta = new std::vector<float>;
  m_allstub_matchTP_phi = new std::vector<float>;

  m_allstub_genuine = new std::vector<int>;

  m_jet_eta = new std::vector<float>;
  m_jet_phi = new std::vector<float>;
  m_jet_pt = new std::vector<float>;
  m_jet_tp_sumpt = new std::vector<float>;
  m_jet_trk_sumpt = new std::vector<float>;
  m_jet_matchtrk_sumpt = new std::vector<float>;

  // ntuple
  eventTree = fs->make<TTree>("eventTree", "Event tree");

  if (SaveAllTracks) {
    eventTree->Branch("trk_pt", &m_trk_pt);
    eventTree->Branch("trk_eta", &m_trk_eta);
    eventTree->Branch("trk_phi", &m_trk_phi);
    eventTree->Branch("trk_d0", &m_trk_d0);
    eventTree->Branch("trk_z0", &m_trk_z0);
    eventTree->Branch("trk_chi2", &m_trk_chi2);
    eventTree->Branch("trk_chi2rphi", &m_trk_chi2rphi);
    eventTree->Branch("trk_chi2rz", &m_trk_chi2rz);
    eventTree->Branch("trk_bendchi2", &m_trk_bendchi2);
    eventTree->Branch("trk_nstub", &m_trk_nstub);
    eventTree->Branch("trk_lhits", &m_trk_lhits);
    eventTree->Branch("trk_dhits", &m_trk_dhits);
    eventTree->Branch("trk_seed", &m_trk_seed);
    eventTree->Branch("trk_hitpattern", &m_trk_hitpattern);
    eventTree->Branch("trk_lhits_hitpattern", &m_trk_lhits_hitpattern);
    eventTree->Branch("trk_dhits_hitpattern", &m_trk_dhits_hitpattern);
    eventTree->Branch("trk_nPSstub_hitpattern", &m_trk_nPSstub_hitpattern);
    eventTree->Branch("trk_n2Sstub_hitpattern", &m_trk_n2Sstub_hitpattern);
    eventTree->Branch("trk_nLostPSstub_hitpattern", &m_trk_nLostPSstub_hitpattern);
    eventTree->Branch("trk_nLost2Sstub_hitpattern", &m_trk_nLost2Sstub_hitpattern);
    eventTree->Branch("trk_nLoststub_V1_hitpattern", &m_trk_nLoststub_V1_hitpattern);
    eventTree->Branch("trk_nLoststub_V2_hitpattern", &m_trk_nLoststub_V2_hitpattern);
    eventTree->Branch("trk_phiSector", &m_trk_phiSector);
    eventTree->Branch("trk_etaSector", &m_trk_etaSector);
    eventTree->Branch("trk_genuine", &m_trk_genuine);
    eventTree->Branch("trk_loose", &m_trk_loose);
    eventTree->Branch("trk_unknown", &m_trk_unknown);
    eventTree->Branch("trk_combinatoric", &m_trk_combinatoric);
    eventTree->Branch("trk_fake", &m_trk_fake);
    eventTree->Branch("trk_MVA1", &m_trk_MVA1);
    eventTree->Branch("trk_matchtp_pdgid", &m_trk_matchtp_pdgid);
    eventTree->Branch("trk_matchtp_pt", &m_trk_matchtp_pt);
    eventTree->Branch("trk_matchtp_eta", &m_trk_matchtp_eta);
    eventTree->Branch("trk_matchtp_phi", &m_trk_matchtp_phi);
    eventTree->Branch("trk_matchtp_z0", &m_trk_matchtp_z0);
    eventTree->Branch("trk_matchtp_dxy", &m_trk_matchtp_dxy);
    eventTree->Branch("trk_matchtp_d0", &m_trk_matchtp_d0);
    if (TrackingInJets) {
      eventTree->Branch("trk_injet", &m_trk_injet);
      eventTree->Branch("trk_injet_highpt", &m_trk_injet_highpt);
      eventTree->Branch("trk_injet_vhighpt", &m_trk_injet_vhighpt);
    }
  }

  eventTree->Branch("tp_pt", &m_tp_pt);
  eventTree->Branch("tp_eta", &m_tp_eta);
  eventTree->Branch("tp_phi", &m_tp_phi);
  eventTree->Branch("tp_dxy", &m_tp_dxy);
  eventTree->Branch("tp_d0", &m_tp_d0);
  eventTree->Branch("tp_z0", &m_tp_z0);
  eventTree->Branch("tp_d0_prod", &m_tp_d0_prod);
  eventTree->Branch("tp_z0_prod", &m_tp_z0_prod);
  eventTree->Branch("tp_pdgid", &m_tp_pdgid);
  eventTree->Branch("tp_nmatch", &m_tp_nmatch);
  eventTree->Branch("tp_nstub", &m_tp_nstub);
  eventTree->Branch("tp_eventid", &m_tp_eventid);
  eventTree->Branch("tp_charge", &m_tp_charge);
  if (TrackingInJets) {
    eventTree->Branch("tp_injet", &m_tp_injet);
    eventTree->Branch("tp_injet_highpt", &m_tp_injet_highpt);
    eventTree->Branch("tp_injet_vhighpt", &m_tp_injet_vhighpt);
  }

  eventTree->Branch("matchtrk_pt", &m_matchtrk_pt);
  eventTree->Branch("matchtrk_eta", &m_matchtrk_eta);
  eventTree->Branch("matchtrk_phi", &m_matchtrk_phi);
  eventTree->Branch("matchtrk_z0", &m_matchtrk_z0);
  eventTree->Branch("matchtrk_d0", &m_matchtrk_d0);
  eventTree->Branch("matchtrk_chi2", &m_matchtrk_chi2);
  eventTree->Branch("matchtrk_chi2rphi", &m_matchtrk_chi2rphi);
  eventTree->Branch("matchtrk_chi2rz", &m_matchtrk_chi2rz);
  eventTree->Branch("matchtrk_bendchi2", &m_matchtrk_bendchi2);
  eventTree->Branch("matchtrk_MVA1", &m_matchtrk_MVA1);
  eventTree->Branch("matchtrk_nstub", &m_matchtrk_nstub);
  eventTree->Branch("matchtrk_lhits", &m_matchtrk_lhits);
  eventTree->Branch("matchtrk_dhits", &m_matchtrk_dhits);
  eventTree->Branch("matchtrk_seed", &m_matchtrk_seed);
  eventTree->Branch("matchtrk_hitpattern", &m_matchtrk_hitpattern);
  if (TrackingInJets) {
    eventTree->Branch("matchtrk_injet", &m_matchtrk_injet);
    eventTree->Branch("matchtrk_injet_highpt", &m_matchtrk_injet_highpt);
    eventTree->Branch("matchtrk_injet_vhighpt", &m_matchtrk_injet_vhighpt);
  }

  if (SaveStubs) {
    eventTree->Branch("allstub_x", &m_allstub_x);
    eventTree->Branch("allstub_y", &m_allstub_y);
    eventTree->Branch("allstub_z", &m_allstub_z);

    eventTree->Branch("allstub_isBarrel", &m_allstub_isBarrel);
    eventTree->Branch("allstub_layer", &m_allstub_layer);
    eventTree->Branch("allstub_isPSmodule", &m_allstub_isPSmodule);

    eventTree->Branch("allstub_trigDisplace", &m_allstub_trigDisplace);
    eventTree->Branch("allstub_trigOffset", &m_allstub_trigOffset);
    eventTree->Branch("allstub_trigPos", &m_allstub_trigPos);
    eventTree->Branch("allstub_trigBend", &m_allstub_trigBend);

    eventTree->Branch("allstub_matchTP_pdgid", &m_allstub_matchTP_pdgid);
    eventTree->Branch("allstub_matchTP_pt", &m_allstub_matchTP_pt);
    eventTree->Branch("allstub_matchTP_eta", &m_allstub_matchTP_eta);
    eventTree->Branch("allstub_matchTP_phi", &m_allstub_matchTP_phi);

    eventTree->Branch("allstub_genuine", &m_allstub_genuine);
  }

  if (TrackingInJets) {
    eventTree->Branch("jet_eta", &m_jet_eta);
    eventTree->Branch("jet_phi", &m_jet_phi);
    eventTree->Branch("jet_pt", &m_jet_pt);
    eventTree->Branch("jet_tp_sumpt", &m_jet_tp_sumpt);
    eventTree->Branch("jet_trk_sumpt", &m_jet_trk_sumpt);
    eventTree->Branch("jet_matchtrk_sumpt", &m_jet_matchtrk_sumpt);
  }
}

//////////
// ANALYZE
void L1TrackNtupleMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (not available_)
    return;  // No ROOT file open.

  if (!(MyProcess == 13 || MyProcess == 11 || MyProcess == 211 || MyProcess == 6 || MyProcess == 15 ||
        MyProcess == 1)) {
    edm::LogVerbatim("Tracklet") << "The specified MyProcess is invalid! Exiting...";
    return;
  }

  if (!(L1Tk_nPar == 4 || L1Tk_nPar == 5)) {
    edm::LogVerbatim("Tracklet") << "Invalid number of track parameters, specified L1Tk_nPar == " << L1Tk_nPar
                                 << " but only 4/5 are valid options! Exiting...";
    return;
  }

  // clear variables
  if (SaveAllTracks) {
    m_trk_pt->clear();
    m_trk_eta->clear();
    m_trk_phi->clear();
    m_trk_d0->clear();
    m_trk_z0->clear();
    m_trk_chi2->clear();
    m_trk_chi2rphi->clear();
    m_trk_chi2rz->clear();
    m_trk_bendchi2->clear();
    m_trk_nstub->clear();
    m_trk_lhits->clear();
    m_trk_dhits->clear();
    m_trk_seed->clear();
    m_trk_hitpattern->clear();
    m_trk_lhits_hitpattern->clear();
    m_trk_dhits_hitpattern->clear();
    m_trk_nPSstub_hitpattern->clear();
    m_trk_n2Sstub_hitpattern->clear();
    m_trk_nLostPSstub_hitpattern->clear();
    m_trk_nLost2Sstub_hitpattern->clear();
    m_trk_nLoststub_V1_hitpattern->clear();
    m_trk_nLoststub_V2_hitpattern->clear();
    m_trk_phiSector->clear();
    m_trk_etaSector->clear();
    m_trk_genuine->clear();
    m_trk_loose->clear();
    m_trk_unknown->clear();
    m_trk_combinatoric->clear();
    m_trk_fake->clear();
    m_trk_MVA1->clear();
    m_trk_matchtp_pdgid->clear();
    m_trk_matchtp_pt->clear();
    m_trk_matchtp_eta->clear();
    m_trk_matchtp_phi->clear();
    m_trk_matchtp_z0->clear();
    m_trk_matchtp_dxy->clear();
    m_trk_matchtp_d0->clear();
    m_trk_injet->clear();
    m_trk_injet_highpt->clear();
    m_trk_injet_vhighpt->clear();
  }

  m_tp_pt->clear();
  m_tp_eta->clear();
  m_tp_phi->clear();
  m_tp_dxy->clear();
  m_tp_d0->clear();
  m_tp_z0->clear();
  m_tp_d0_prod->clear();
  m_tp_z0_prod->clear();
  m_tp_pdgid->clear();
  m_tp_nmatch->clear();
  m_tp_nstub->clear();
  m_tp_eventid->clear();
  m_tp_charge->clear();
  m_tp_injet->clear();
  m_tp_injet_highpt->clear();
  m_tp_injet_vhighpt->clear();

  m_matchtrk_pt->clear();
  m_matchtrk_eta->clear();
  m_matchtrk_phi->clear();
  m_matchtrk_z0->clear();
  m_matchtrk_d0->clear();
  m_matchtrk_chi2->clear();
  m_matchtrk_chi2rphi->clear();
  m_matchtrk_chi2rz->clear();
  m_matchtrk_bendchi2->clear();
  m_matchtrk_MVA1->clear();
  m_matchtrk_nstub->clear();
  m_matchtrk_lhits->clear();
  m_matchtrk_dhits->clear();
  m_matchtrk_seed->clear();
  m_matchtrk_hitpattern->clear();
  m_matchtrk_injet->clear();
  m_matchtrk_injet_highpt->clear();
  m_matchtrk_injet_vhighpt->clear();

  if (SaveStubs) {
    m_allstub_x->clear();
    m_allstub_y->clear();
    m_allstub_z->clear();

    m_allstub_isBarrel->clear();
    m_allstub_layer->clear();
    m_allstub_isPSmodule->clear();

    m_allstub_trigDisplace->clear();
    m_allstub_trigOffset->clear();
    m_allstub_trigPos->clear();
    m_allstub_trigBend->clear();

    m_allstub_matchTP_pdgid->clear();
    m_allstub_matchTP_pt->clear();
    m_allstub_matchTP_eta->clear();
    m_allstub_matchTP_phi->clear();

    m_allstub_genuine->clear();
  }

  m_jet_eta->clear();
  m_jet_phi->clear();
  m_jet_pt->clear();
  m_jet_tp_sumpt->clear();
  m_jet_trk_sumpt->clear();
  m_jet_matchtrk_sumpt->clear();

  // -----------------------------------------------------------------------------------------------
  // retrieve various containers
  // -----------------------------------------------------------------------------------------------

  // L1 tracks
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > TTTrackHandle;
  iEvent.getByToken(ttTrackToken_, TTTrackHandle);

  // L1 stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> > > TTStubHandle;
  if (SaveStubs)
    iEvent.getByToken(ttStubToken_, TTStubHandle);

  // MC truth association maps
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_> > MCTruthTTClusterHandle;
  iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_> > MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > MCTruthTTTrackHandle;
  iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);

  // tracking particles
  edm::Handle<std::vector<TrackingParticle> > TrackingParticleHandle;
  edm::Handle<std::vector<TrackingVertex> > TrackingVertexHandle;
  iEvent.getByToken(TrackingParticleToken_, TrackingParticleHandle);
  //iEvent.getByToken(TrackingVertexToken_, TrackingVertexHandle);

  // -----------------------------------------------------------------------------------------------
  // more for TTStubs
  edm::ESHandle<TrackerGeometry> tGeomHandle = iSetup.getHandle(getTokenTrackerGeom_);

  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(getTokenTrackerTopo_);

  edm::ESHandle<MagneticField> bFieldHandle = iSetup.getHandle(getTokenBField_);

  edm::ESHandle<hph::Setup> hphHandle = iSetup.getHandle(getTokenHPHSetup_);

  const TrackerTopology* const tTopo = tTopoHandle.product();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();
  const hph::Setup* hphSetup = hphHandle.product();

  // ----------------------------------------------------------------------------------------------
  // loop over L1 stubs
  // ----------------------------------------------------------------------------------------------

  if (SaveStubs) {
    for (auto gd = theTrackerGeom->dets().begin(); gd != theTrackerGeom->dets().end(); gd++) {
      DetId detid = (*gd)->geographicalId();
      if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
        continue;
      if (!tTopo->isLower(detid))
        continue;                              // loop on the stacks: choose the lower arbitrarily
      DetId stackDetid = tTopo->stack(detid);  // Stub module detid

      if (TTStubHandle->find(stackDetid) == TTStubHandle->end())
        continue;

      // Get the DetSets of the Clusters
      edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_> > stubs = (*TTStubHandle)[stackDetid];
      const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit(detid);
      const auto* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(det0);
      const PixelTopology* topol = dynamic_cast<const PixelTopology*>(&(theGeomDet->specificTopology()));

      // loop over stubs
      for (auto stubIter = stubs.begin(); stubIter != stubs.end(); ++stubIter) {
        edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > tempStubPtr =
            edmNew::makeRefTo(TTStubHandle, stubIter);

        int isBarrel = 0;
        int layer = -999999;
        if (detid.subdetId() == StripSubdetector::TOB) {
          isBarrel = 1;
          layer = static_cast<int>(tTopo->layer(detid));
        } else if (detid.subdetId() == StripSubdetector::TID) {
          isBarrel = 0;
          layer = static_cast<int>(tTopo->layer(detid));
        } else {
          edm::LogVerbatim("Tracklet") << "WARNING -- neither TOB or TID stub, shouldn't happen...";
          layer = -1;
        }

        int isPSmodule = 0;
        if (topol->nrows() == 960)
          isPSmodule = 1;

        MeasurementPoint coords = tempStubPtr->clusterRef(0)->findAverageLocalCoordinatesCentered();
        LocalPoint clustlp = topol->localPosition(coords);
        GlobalPoint posStub = theGeomDet->surface().toGlobal(clustlp);

        double tmp_stub_x = posStub.x();
        double tmp_stub_y = posStub.y();
        double tmp_stub_z = posStub.z();

        float trigDisplace = tempStubPtr->rawBend();
        float trigOffset = tempStubPtr->bendOffset();
        float trigPos = tempStubPtr->innerClusterPosition();
        float trigBend = tempStubPtr->bendFE();

        m_allstub_x->push_back(tmp_stub_x);
        m_allstub_y->push_back(tmp_stub_y);
        m_allstub_z->push_back(tmp_stub_z);

        m_allstub_isBarrel->push_back(isBarrel);
        m_allstub_layer->push_back(layer);
        m_allstub_isPSmodule->push_back(isPSmodule);

        m_allstub_trigDisplace->push_back(trigDisplace);
        m_allstub_trigOffset->push_back(trigOffset);
        m_allstub_trigPos->push_back(trigPos);
        m_allstub_trigBend->push_back(trigBend);

        // matched to tracking particle?
        edm::Ptr<TrackingParticle> my_tp = MCTruthTTStubHandle->findTrackingParticlePtr(tempStubPtr);

        int myTP_pdgid = -999;
        float myTP_pt = -999;
        float myTP_eta = -999;
        float myTP_phi = -999;

        if (my_tp.isNull() == false) {
          int tmp_eventid = my_tp->eventId().event();

          if (tmp_eventid > 0)
            continue;  // this means stub from pileup track

          myTP_pdgid = my_tp->pdgId();
          myTP_pt = my_tp->p4().pt();
          myTP_eta = my_tp->p4().eta();
          myTP_phi = my_tp->p4().phi();
        }

        m_allstub_matchTP_pdgid->push_back(myTP_pdgid);
        m_allstub_matchTP_pt->push_back(myTP_pt);
        m_allstub_matchTP_eta->push_back(myTP_eta);
        m_allstub_matchTP_phi->push_back(myTP_phi);

        int tmp_stub_genuine = 0;
        if (MCTruthTTStubHandle->isGenuine(tempStubPtr))
          tmp_stub_genuine = 1;

        m_allstub_genuine->push_back(tmp_stub_genuine);
      }
    }
  }

  // ----------------------------------------------------------------------------------------------
  // tracking in jets
  // ----------------------------------------------------------------------------------------------

  std::vector<math::XYZTLorentzVector> v_jets;
  std::vector<int> v_jets_highpt;
  std::vector<int> v_jets_vhighpt;

  if (TrackingInJets) {
    // gen jets
    if (DebugMode)
      edm::LogVerbatim("Tracklet") << "get genjets";
    edm::Handle<std::vector<reco::GenJet> > GenJetHandle;
    iEvent.getByToken(GenJetToken_, GenJetHandle);

    if (GenJetHandle.isValid()) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "loop over genjets";
      std::vector<reco::GenJet>::const_iterator iterGenJet;
      for (iterGenJet = GenJetHandle->begin(); iterGenJet != GenJetHandle->end(); ++iterGenJet) {
        reco::GenJet myJet = reco::GenJet(*iterGenJet);

        if (myJet.pt() < 30.0)
          continue;
        if (std::abs(myJet.eta()) > 2.5)
          continue;

        if (DebugMode)
          edm::LogVerbatim("Tracklet") << "genjet pt = " << myJet.pt() << ", eta = " << myJet.eta();

        bool ishighpt = false;
        bool isveryhighpt = false;
        if (myJet.pt() > 100.0)
          ishighpt = true;
        if (myJet.pt() > 200.0)
          isveryhighpt = true;

        const math::XYZTLorentzVector& jetP4 = myJet.p4();
        v_jets.push_back(jetP4);
        if (ishighpt)
          v_jets_highpt.push_back(1);
        else
          v_jets_highpt.push_back(0);
        if (isveryhighpt)
          v_jets_vhighpt.push_back(1);
        else
          v_jets_vhighpt.push_back(0);

      }  // end loop over genjets
    }    // end isValid

  }  // end TrackingInJets

  const int NJETS = 10;
  float jets_tp_sumpt[NJETS] = {0};        //sum pt of TPs with dR<0.4 of jet
  float jets_matchtrk_sumpt[NJETS] = {0};  //sum pt of tracks matched to TP with dR<0.4 of jet
  float jets_trk_sumpt[NJETS] = {0};       //sum pt of all tracks with dR<0.4 of jet

  // ----------------------------------------------------------------------------------------------
  // loop over L1 tracks
  // ----------------------------------------------------------------------------------------------

  if (SaveAllTracks) {
    if (DebugMode) {
      edm::LogVerbatim("Tracklet") << "\n Loop over L1 tracks!";
      edm::LogVerbatim("Tracklet") << "\n Looking at " << L1Tk_nPar << "-parameter tracks!";
    }

    int this_l1track = 0;
    std::vector<TTTrack<Ref_Phase2TrackerDigi_> >::const_iterator iterL1Track;
    for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
      edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > l1track_ptr(TTTrackHandle, this_l1track);
      this_l1track++;

      float tmp_trk_pt = iterL1Track->momentum().perp();
      float tmp_trk_eta = iterL1Track->momentum().eta();
      float tmp_trk_phi = iterL1Track->momentum().phi();
      float tmp_trk_z0 = iterL1Track->z0();  //cm
      float tmp_trk_tanL = iterL1Track->tanL();

      int tmp_trk_hitpattern = 0;
      tmp_trk_hitpattern = (int)iterL1Track->hitPattern();
      hph::HitPatternHelper hph(hphSetup, tmp_trk_hitpattern, tmp_trk_tanL, tmp_trk_z0);
      std::vector<int> hitpattern_expanded_binary = hph.binary();
      int tmp_trk_lhits_hitpattern = 0;
      int tmp_trk_dhits_hitpattern = 0;
      for (int i = 0; i < (int)hitpattern_expanded_binary.size(); i++) {
        if (hitpattern_expanded_binary[i]) {
          if (i < 6) {
            tmp_trk_lhits_hitpattern += pow(10, i);
          } else {
            tmp_trk_dhits_hitpattern += pow(10, i - 6);
          }
        }
      }
      int tmp_trk_nPSstub_hitpattern = hph.numPS();
      int tmp_trk_n2Sstub_hitpattern = hph.num2S();
      int tmp_trk_nLostPSstub_hitpattern = hph.numMissingPS();
      int tmp_trk_nLost2Sstub_hitpattern = hph.numMissing2S();
      int tmp_trk_nLoststub_V1_hitpattern = hph.numMissingInterior1();
      int tmp_trk_nLoststub_V2_hitpattern = hph.numMissingInterior2();

      float tmp_trk_d0 = -999;
      if (L1Tk_nPar == 5) {
        float tmp_trk_x0 = iterL1Track->POCA().x();
        float tmp_trk_y0 = iterL1Track->POCA().y();
        tmp_trk_d0 = tmp_trk_x0 * sin(tmp_trk_phi) - tmp_trk_y0 * cos(tmp_trk_phi);
      }

      float tmp_trk_chi2 = iterL1Track->chi2();
      float tmp_trk_chi2rphi = iterL1Track->chi2XY();
      float tmp_trk_chi2rz = iterL1Track->chi2Z();
      float tmp_trk_bendchi2 = iterL1Track->stubPtConsistency();
      float tmp_trk_MVA1 = iterL1Track->trkMVA1();

      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
          stubRefs = iterL1Track->getStubRefs();
      int tmp_trk_nstub = (int)stubRefs.size();

      int tmp_trk_seed = 0;
      tmp_trk_seed = (int)iterL1Track->trackSeedType();

      unsigned int tmp_trk_phiSector = iterL1Track->phiSector();
      int tmp_trk_etaSector = hph.etaSector();

      // ----------------------------------------------------------------------------------------------
      // loop over stubs on tracks

      //float tmp_trk_bend_chi2 = 0;
      int tmp_trk_dhits = 0;
      int tmp_trk_lhits = 0;

      if (true) {
        // loop over stubs
        for (int is = 0; is < tmp_trk_nstub; is++) {
          //detID of stub
          DetId detIdStub = theTrackerGeom->idToDet((stubRefs.at(is)->clusterRef(0))->getDetId())->geographicalId();

          MeasurementPoint coords = stubRefs.at(is)->clusterRef(0)->findAverageLocalCoordinatesCentered();
          const GeomDet* theGeomDet = theTrackerGeom->idToDet(detIdStub);
          Global3DPoint posStub = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(coords));

          double x = posStub.x();
          double y = posStub.y();
          double z = posStub.z();

          int layer = -999999;
          if (detIdStub.subdetId() == StripSubdetector::TOB) {
            layer = static_cast<int>(tTopo->layer(detIdStub));
            if (DebugMode)
              edm::LogVerbatim("Tracklet")
                  << "   stub in layer " << layer << " at position x y z = " << x << " " << y << " " << z;
            tmp_trk_lhits += pow(10, layer - 1);
          } else if (detIdStub.subdetId() == StripSubdetector::TID) {
            layer = static_cast<int>(tTopo->layer(detIdStub));
            if (DebugMode)
              edm::LogVerbatim("Tracklet")
                  << "   stub in disk " << layer << " at position x y z = " << x << " " << y << " " << z;
            tmp_trk_dhits += pow(10, layer - 1);
          }

        }  //end loop over stubs
      }
      // ----------------------------------------------------------------------------------------------

      int tmp_trk_genuine = 0;
      int tmp_trk_loose = 0;
      int tmp_trk_unknown = 0;
      int tmp_trk_combinatoric = 0;
      if (MCTruthTTTrackHandle->isLooselyGenuine(l1track_ptr))
        tmp_trk_loose = 1;
      if (MCTruthTTTrackHandle->isGenuine(l1track_ptr))
        tmp_trk_genuine = 1;
      if (MCTruthTTTrackHandle->isUnknown(l1track_ptr))
        tmp_trk_unknown = 1;
      if (MCTruthTTTrackHandle->isCombinatoric(l1track_ptr))
        tmp_trk_combinatoric = 1;

      if (DebugMode) {
        edm::LogVerbatim("Tracklet") << "L1 track,"
                                     << " pt: " << tmp_trk_pt << " eta: " << tmp_trk_eta << " phi: " << tmp_trk_phi
                                     << " z0: " << tmp_trk_z0 << " chi2: " << tmp_trk_chi2
                                     << " chi2rphi: " << tmp_trk_chi2rphi << " chi2rz: " << tmp_trk_chi2rz
                                     << " nstub: " << tmp_trk_nstub;
        if (tmp_trk_genuine)
          edm::LogVerbatim("Tracklet") << "    (is genuine)";
        if (tmp_trk_unknown)
          edm::LogVerbatim("Tracklet") << "    (is unknown)";
        if (tmp_trk_combinatoric)
          edm::LogVerbatim("Tracklet") << "    (is combinatoric)";
      }

      m_trk_pt->push_back(tmp_trk_pt);
      m_trk_eta->push_back(tmp_trk_eta);
      m_trk_phi->push_back(tmp_trk_phi);
      m_trk_z0->push_back(tmp_trk_z0);
      if (L1Tk_nPar == 5)
        m_trk_d0->push_back(tmp_trk_d0);
      else
        m_trk_d0->push_back(999.);
      m_trk_chi2->push_back(tmp_trk_chi2);
      m_trk_chi2rphi->push_back(tmp_trk_chi2rphi);
      m_trk_chi2rz->push_back(tmp_trk_chi2rz);
      m_trk_bendchi2->push_back(tmp_trk_bendchi2);
      m_trk_MVA1->push_back(tmp_trk_MVA1);
      m_trk_nstub->push_back(tmp_trk_nstub);
      m_trk_dhits->push_back(tmp_trk_dhits);
      m_trk_lhits->push_back(tmp_trk_lhits);
      m_trk_seed->push_back(tmp_trk_seed);
      m_trk_hitpattern->push_back(tmp_trk_hitpattern);
      m_trk_lhits_hitpattern->push_back(tmp_trk_lhits_hitpattern);
      m_trk_dhits_hitpattern->push_back(tmp_trk_dhits_hitpattern);
      m_trk_nPSstub_hitpattern->push_back(tmp_trk_nPSstub_hitpattern);
      m_trk_n2Sstub_hitpattern->push_back(tmp_trk_n2Sstub_hitpattern);
      m_trk_nLostPSstub_hitpattern->push_back(tmp_trk_nLostPSstub_hitpattern);
      m_trk_nLost2Sstub_hitpattern->push_back(tmp_trk_nLost2Sstub_hitpattern);
      m_trk_nLoststub_V1_hitpattern->push_back(tmp_trk_nLoststub_V1_hitpattern);
      m_trk_nLoststub_V2_hitpattern->push_back(tmp_trk_nLoststub_V2_hitpattern);
      m_trk_phiSector->push_back(tmp_trk_phiSector);
      m_trk_etaSector->push_back(tmp_trk_etaSector);
      m_trk_genuine->push_back(tmp_trk_genuine);
      m_trk_loose->push_back(tmp_trk_loose);
      m_trk_unknown->push_back(tmp_trk_unknown);
      m_trk_combinatoric->push_back(tmp_trk_combinatoric);

      // ----------------------------------------------------------------------------------------------
      // for studying the fake rate
      // ----------------------------------------------------------------------------------------------

      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);

      int myFake = 0;

      int tmp_matchtp_pdgid = -999;
      float tmp_matchtp_pt = -999;
      float tmp_matchtp_eta = -999;
      float tmp_matchtp_phi = -999;
      float tmp_matchtp_z0 = -999;
      float tmp_matchtp_dxy = -999;
      float tmp_matchtp_d0 = -999;

      if (my_tp.isNull())
        myFake = 0;
      else {
        int tmp_eventid = my_tp->eventId().event();

        if (tmp_eventid > 0)
          myFake = 2;
        else
          myFake = 1;

        tmp_matchtp_pdgid = my_tp->pdgId();
        tmp_matchtp_pt = my_tp->pt();
        tmp_matchtp_eta = my_tp->eta();
        tmp_matchtp_phi = my_tp->phi();

        float tmp_matchtp_vz = my_tp->vz();
        float tmp_matchtp_vx = my_tp->vx();
        float tmp_matchtp_vy = my_tp->vy();
        tmp_matchtp_dxy = sqrt(tmp_matchtp_vx * tmp_matchtp_vx + tmp_matchtp_vy * tmp_matchtp_vy);

        // ----------------------------------------------------------------------------------------------
        // get d0/z0 propagated back to the IP

        float tmp_matchtp_t = 1.0 / tan(2.0 * atan(exp(-tmp_matchtp_eta)));

        float delx = -tmp_matchtp_vx;
        float dely = -tmp_matchtp_vy;

        float b_field = bFieldHandle.product()->inTesla(GlobalPoint(0, 0, 0)).z();
        float c_converted = CLHEP::c_light / 1.0E5;
        float r2_inv = my_tp->charge() * c_converted * b_field / tmp_matchtp_pt / 2.0;

        float tmp_matchtp_x0p = delx - (1. / (2. * r2_inv) * sin(tmp_matchtp_phi));
        float tmp_matchtp_y0p = dely + (1. / (2. * r2_inv) * cos(tmp_matchtp_phi));
        float tmp_matchtp_rp = sqrt(tmp_matchtp_x0p * tmp_matchtp_x0p + tmp_matchtp_y0p * tmp_matchtp_y0p);
        tmp_matchtp_d0 = my_tp->charge() * tmp_matchtp_rp - (1. / (2. * r2_inv));

        static double pi = M_PI;
        float delphi = tmp_matchtp_phi - atan2(-r2_inv * tmp_matchtp_x0p, r2_inv * tmp_matchtp_y0p);
        if (delphi < -pi)
          delphi += 2.0 * pi;
        if (delphi > pi)
          delphi -= 2.0 * pi;
        tmp_matchtp_z0 = tmp_matchtp_vz + tmp_matchtp_t * delphi / (2.0 * r2_inv);
        // ----------------------------------------------------------------------------------------------

        if (DebugMode) {
          edm::LogVerbatim("Tracklet") << "TP matched to track has pt = " << my_tp->p4().pt()
                                       << " eta = " << my_tp->momentum().eta() << " phi = " << my_tp->momentum().phi()
                                       << " z0 = " << my_tp->vertex().z() << " pdgid = " << my_tp->pdgId()
                                       << " dxy = " << tmp_matchtp_dxy;
        }
      }

      m_trk_fake->push_back(myFake);

      m_trk_matchtp_pdgid->push_back(tmp_matchtp_pdgid);
      m_trk_matchtp_pt->push_back(tmp_matchtp_pt);
      m_trk_matchtp_eta->push_back(tmp_matchtp_eta);
      m_trk_matchtp_phi->push_back(tmp_matchtp_phi);
      m_trk_matchtp_z0->push_back(tmp_matchtp_z0);
      m_trk_matchtp_dxy->push_back(tmp_matchtp_dxy);
      m_trk_matchtp_d0->push_back(tmp_matchtp_d0);

      // ----------------------------------------------------------------------------------------------
      // for tracking in jets
      // ----------------------------------------------------------------------------------------------

      if (TrackingInJets) {
        if (DebugMode)
          edm::LogVerbatim("Tracklet") << "doing tracking in jets now";

        int InJet = 0;
        int InJetHighpt = 0;
        int InJetVeryHighpt = 0;

        for (int ij = 0; ij < (int)v_jets.size(); ij++) {
          float deta = tmp_trk_eta - (v_jets.at(ij)).eta();
          float dphi = tmp_trk_phi - (v_jets.at(ij)).phi();
          while (dphi > 3.14159)
            dphi = std::abs(2 * 3.14159 - dphi);
          float dR = sqrt(deta * deta + dphi * dphi);

          if (dR < 0.4) {
            InJet = 1;
            if (v_jets_highpt.at(ij) == 1)
              InJetHighpt = 1;
            if (v_jets_vhighpt.at(ij) == 1)
              InJetVeryHighpt = 1;
            if (ij < NJETS)
              jets_trk_sumpt[ij] += tmp_trk_pt;
          }
        }

        m_trk_injet->push_back(InJet);
        m_trk_injet_highpt->push_back(InJetHighpt);
        m_trk_injet_vhighpt->push_back(InJetVeryHighpt);

      }  //end tracking in jets

    }  //end track loop

  }  //end if SaveAllTracks

  // ----------------------------------------------------------------------------------------------
  // loop over tracking particles
  // ----------------------------------------------------------------------------------------------

  if (DebugMode)
    edm::LogVerbatim("Tracklet") << "\n Loop over tracking particles!";

  int this_tp = 0;
  std::vector<TrackingParticle>::const_iterator iterTP;
  for (iterTP = TrackingParticleHandle->begin(); iterTP != TrackingParticleHandle->end(); ++iterTP) {
    edm::Ptr<TrackingParticle> tp_ptr(TrackingParticleHandle, this_tp);
    this_tp++;

    int tmp_eventid = iterTP->eventId().event();
    if (MyProcess != 1 && tmp_eventid > 0)
      continue;  //only care about tracking particles from the primary interaction (except for MyProcess==1, i.e. looking at all TPs)

    float tmp_tp_pt = iterTP->pt();
    float tmp_tp_charge = iterTP->charge();
    float tmp_tp_eta = iterTP->eta();

    if (tmp_tp_pt < TP_minPt)  // Save CPU by applying these cuts here.
      continue;
    if (tmp_tp_charge == 0.)
      continue;
    if (std::abs(tmp_tp_eta) > TP_maxEta)
      continue;

    float tmp_tp_phi = iterTP->phi();
    float tmp_tp_vz = iterTP->vz();
    float tmp_tp_vx = iterTP->vx();
    float tmp_tp_vy = iterTP->vy();
    int tmp_tp_pdgid = iterTP->pdgId();
    float tmp_tp_z0_prod = tmp_tp_vz;
    float tmp_tp_d0_prod = tmp_tp_vx * sin(tmp_tp_phi) - tmp_tp_vy * cos(tmp_tp_phi);

    // ----------------------------------------------------------------------------------------------
    // get d0/z0 propagated back to the IP

    float tmp_tp_t = 1.0 / tan(2.0 * atan(exp(-tmp_tp_eta)));

    float delx = -tmp_tp_vx;
    float dely = -tmp_tp_vy;

    float b_field = bFieldHandle.product()->inTesla(GlobalPoint(0, 0, 0)).z();
    float c_converted = CLHEP::c_light / 1.0E5;
    float r2_inv = tmp_tp_charge * c_converted * b_field / tmp_tp_pt / 2.0;

    float tmp_tp_x0p = delx - (1. / (2. * r2_inv) * sin(tmp_tp_phi));
    float tmp_tp_y0p = dely + (1. / (2. * r2_inv) * cos(tmp_tp_phi));
    float tmp_tp_rp = sqrt(tmp_tp_x0p * tmp_tp_x0p + tmp_tp_y0p * tmp_tp_y0p);
    float tmp_tp_d0 = tmp_tp_charge * tmp_tp_rp - (1. / (2. * r2_inv));

    static double pi = M_PI;
    float delphi = tmp_tp_phi - atan2(-r2_inv * tmp_tp_x0p, r2_inv * tmp_tp_y0p);
    if (delphi < -pi)
      delphi += 2.0 * pi;
    if (delphi > pi)
      delphi -= 2.0 * pi;
    float tmp_tp_z0 = tmp_tp_vz + tmp_tp_t * delphi / (2.0 * r2_inv);
    // ----------------------------------------------------------------------------------------------

    if (MyProcess == 13 && abs(tmp_tp_pdgid) != 13)
      continue;
    if (MyProcess == 11 && abs(tmp_tp_pdgid) != 11)
      continue;
    if ((MyProcess == 6 || MyProcess == 15 || MyProcess == 211) && abs(tmp_tp_pdgid) != 211)
      continue;

    if (std::abs(tmp_tp_z0) > TP_maxZ0)
      continue;

    // for pions in ttbar, only consider TPs coming from near the IP!
    float dxy = sqrt(tmp_tp_vx * tmp_tp_vx + tmp_tp_vy * tmp_tp_vy);
    float tmp_tp_dxy = dxy;
    if (MyProcess == 6 && (dxy > 1.0))
      continue;

    if (DebugMode)
      edm::LogVerbatim("Tracklet") << "Tracking particle, pt: " << tmp_tp_pt << " eta: " << tmp_tp_eta
                                   << " phi: " << tmp_tp_phi << " z0: " << tmp_tp_z0 << " d0: " << tmp_tp_d0
                                   << " z_prod: " << tmp_tp_z0_prod << " d_prod: " << tmp_tp_d0_prod
                                   << " pdgid: " << tmp_tp_pdgid << " eventID: " << iterTP->eventId().event()
                                   << " ttclusters " << MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).size()
                                   << " ttstubs " << MCTruthTTStubHandle->findTTStubRefs(tp_ptr).size() << " tttracks "
                                   << MCTruthTTTrackHandle->findTTTrackPtrs(tp_ptr).size();

    // ----------------------------------------------------------------------------------------------
    // only consider TPs associated with >= 1 cluster, or >= X stubs, or have stubs in >= X layers (configurable options)

    if (MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).empty()) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "No matching TTClusters for TP, continuing...";
      continue;
    }

    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
        theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);
    int nStubTP = (int)theStubRefs.size();

    // how many layers/disks have stubs?
    int hasStubInLayer[11] = {0};
    for (auto& theStubRef : theStubRefs) {
      DetId detid(theStubRef->getDetId());

      int layer = -1;
      if (detid.subdetId() == StripSubdetector::TOB) {
        layer = static_cast<int>(tTopo->layer(detid)) - 1;  //fill in array as entries 0-5
      } else if (detid.subdetId() == StripSubdetector::TID) {
        layer = static_cast<int>(tTopo->layer(detid)) + 5;  //fill in array as entries 6-10
      }

      //bool isPS = (theTrackerGeom->getDetectorType(detid)==TrackerGeometry::ModuleType::Ph2PSP);

      //treat genuine stubs separately (==2 is genuine, ==1 is not)
      if (MCTruthTTStubHandle->findTrackingParticlePtr(theStubRef).isNull() && hasStubInLayer[layer] < 2)
        hasStubInLayer[layer] = 1;
      else
        hasStubInLayer[layer] = 2;
    }

    int nStubLayerTP = 0;
    int nStubLayerTP_g = 0;
    for (int isum : hasStubInLayer) {
      if (isum >= 1)
        nStubLayerTP += 1;
      if (isum == 2)
        nStubLayerTP_g += 1;
    }

    if (DebugMode)
      edm::LogVerbatim("Tracklet") << "TP is associated with " << nStubTP << " stubs, and has stubs in " << nStubLayerTP
                                   << " different layers/disks, and has GENUINE stubs in " << nStubLayerTP_g
                                   << " layers ";

    if (TP_minNStub > 0) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "Only consider TPs with >= " << TP_minNStub << " stubs";
      if (nStubTP < TP_minNStub) {
        if (DebugMode)
          edm::LogVerbatim("Tracklet") << "TP fails minimum nbr stubs requirement! Continuing...";
        continue;
      }
    }
    if (TP_minNStubLayer > 0) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "Only consider TPs with stubs in >= " << TP_minNStubLayer << " layers/disks";
      if (nStubLayerTP < TP_minNStubLayer) {
        if (DebugMode)
          edm::LogVerbatim("Tracklet") << "TP fails stubs in minimum nbr of layers/disks requirement! Continuing...";
        continue;
      }
    }

    // ----------------------------------------------------------------------------------------------
    // look for L1 tracks matched to the tracking particle

    std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > > matchedTracks =
        MCTruthTTTrackHandle->findTTTrackPtrs(tp_ptr);

    int nMatch = 0;
    int i_track = -1;
    float i_chi2dof = 99999;

    if (!matchedTracks.empty()) {
      if (DebugMode && (matchedTracks.size() > 1))
        edm::LogVerbatim("Tracklet") << "TrackingParticle has more than one matched L1 track!";

      // ----------------------------------------------------------------------------------------------
      // loop over matched L1 tracks
      // here, "match" means tracks that can be associated to a TrackingParticle with at least one hit of at least one of its clusters
      // https://twiki.cern.ch/twiki/bin/viewauth/CMS/SLHCTrackerTriggerSWTools#MC_truth_for_TTTrack

      for (int it = 0; it < (int)matchedTracks.size(); it++) {
        bool tmp_trk_genuine = false;
        bool tmp_trk_loosegenuine = false;
        if (MCTruthTTTrackHandle->isGenuine(matchedTracks.at(it)))
          tmp_trk_genuine = true;
        if (MCTruthTTTrackHandle->isLooselyGenuine(matchedTracks.at(it)))
          tmp_trk_loosegenuine = true;
        if (!tmp_trk_loosegenuine)
          continue;

        if (DebugMode) {
          if (MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it)).isNull()) {
            edm::LogVerbatim("Tracklet") << "track matched to TP is NOT uniquely matched to a TP";
          } else {
            edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it));
            edm::LogVerbatim("Tracklet") << "TP matched to track matched to TP ... tp pt = " << my_tp->p4().pt()
                                         << " eta = " << my_tp->momentum().eta() << " phi = " << my_tp->momentum().phi()
                                         << " z0 = " << my_tp->vertex().z();
          }
          edm::LogVerbatim("Tracklet") << "   ... matched L1 track has pt = " << matchedTracks.at(it)->momentum().perp()
                                       << " eta = " << matchedTracks.at(it)->momentum().eta()
                                       << " phi = " << matchedTracks.at(it)->momentum().phi()
                                       << " chi2 = " << matchedTracks.at(it)->chi2()
                                       << " consistency = " << matchedTracks.at(it)->stubPtConsistency()
                                       << " z0 = " << matchedTracks.at(it)->z0()
                                       << " nstub = " << matchedTracks.at(it)->getStubRefs().size();
          if (tmp_trk_genuine)
            edm::LogVerbatim("Tracklet") << "    (genuine!) ";
          if (tmp_trk_loosegenuine)
            edm::LogVerbatim("Tracklet") << "    (loose genuine!) ";
        }

        // ----------------------------------------------------------------------------------------------
        // further require L1 track to be (loosely) genuine, that there is only one TP matched to the track
        // + have >= L1Tk_minNStub stubs for it to be a valid match (only relevant is your track collection
        // e.g. stores 3-stub tracks but at plot level you require >= 4 stubs (--> tracklet case)

        std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
            stubRefs = matchedTracks.at(it)->getStubRefs();
        int tmp_trk_nstub = stubRefs.size();

        if (tmp_trk_nstub < L1Tk_minNStub)
          continue;

        /*
	// PS stubs
	int tmp_trk_nPSstub = 0;
	for (int is=0; is<tmp_trk_nstub; is++) {
	  DetId detIdStub = theTrackerGeom->idToDet( (stubRefs.at(is)->clusterRef(0))->getDetId() )->geographicalId();
	  DetId stackDetid = tTopo->stack(detIdStub);
	  bool isPS = (theTrackerGeom->getDetectorType(stackDetid)==TrackerGeometry::ModuleType::Ph2PSP);
	  if (isPS) tmp_trk_nPSstub++;
	}
	*/

        float dmatch_pt = 999;
        float dmatch_eta = 999;
        float dmatch_phi = 999;
        int match_id = 999;

        edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it));
        dmatch_pt = std::abs(my_tp->p4().pt() - tmp_tp_pt);
        dmatch_eta = std::abs(my_tp->p4().eta() - tmp_tp_eta);
        dmatch_phi = std::abs(my_tp->p4().phi() - tmp_tp_phi);
        match_id = my_tp->pdgId();

        float tmp_trk_chi2dof = (matchedTracks.at(it)->chi2()) / (2 * tmp_trk_nstub - L1Tk_nPar);

        // ensure that track is uniquely matched to the TP we are looking at!
        if (dmatch_pt < 0.1 && dmatch_eta < 0.1 && dmatch_phi < 0.1 && tmp_tp_pdgid == match_id && tmp_trk_genuine) {
          nMatch++;
          if (i_track < 0 || tmp_trk_chi2dof < i_chi2dof) {
            i_track = it;
            i_chi2dof = tmp_trk_chi2dof;
          }
        }

      }  // end loop over matched L1 tracks

    }  // end has at least 1 matched L1 track
    // ----------------------------------------------------------------------------------------------

    float tmp_matchtrk_pt = -999;
    float tmp_matchtrk_eta = -999;
    float tmp_matchtrk_phi = -999;
    float tmp_matchtrk_z0 = -999;
    float tmp_matchtrk_d0 = -999;
    float tmp_matchtrk_chi2 = -999;
    float tmp_matchtrk_chi2rphi = -999;
    float tmp_matchtrk_chi2rz = -999;
    float tmp_matchtrk_bendchi2 = -999;
    float tmp_matchtrk_MVA1 = -999;
    int tmp_matchtrk_nstub = -999;
    int tmp_matchtrk_dhits = -999;
    int tmp_matchtrk_lhits = -999;
    int tmp_matchtrk_seed = -999;
    int tmp_matchtrk_hitpattern = -999;

    if (nMatch > 1 && DebugMode)
      edm::LogVerbatim("Tracklet") << "WARNING *** 2 or more matches to genuine L1 tracks ***";

    if (nMatch > 0) {
      tmp_matchtrk_pt = matchedTracks.at(i_track)->momentum().perp();
      tmp_matchtrk_eta = matchedTracks.at(i_track)->momentum().eta();
      tmp_matchtrk_phi = matchedTracks.at(i_track)->momentum().phi();
      tmp_matchtrk_z0 = matchedTracks.at(i_track)->z0();

      if (L1Tk_nPar == 5) {
        float tmp_matchtrk_x0 = matchedTracks.at(i_track)->POCA().x();
        float tmp_matchtrk_y0 = matchedTracks.at(i_track)->POCA().y();
        tmp_matchtrk_d0 = tmp_matchtrk_x0 * sin(tmp_matchtrk_phi) - tmp_matchtrk_y0 * cos(tmp_matchtrk_phi);
      }

      tmp_matchtrk_chi2 = matchedTracks.at(i_track)->chi2();
      tmp_matchtrk_chi2rphi = matchedTracks.at(i_track)->chi2XY();
      tmp_matchtrk_chi2rz = matchedTracks.at(i_track)->chi2Z();
      tmp_matchtrk_bendchi2 = matchedTracks.at(i_track)->stubPtConsistency();
      tmp_matchtrk_MVA1 = matchedTracks.at(i_track)->trkMVA1();
      tmp_matchtrk_nstub = (int)matchedTracks.at(i_track)->getStubRefs().size();
      tmp_matchtrk_seed = (int)matchedTracks.at(i_track)->trackSeedType();
      tmp_matchtrk_hitpattern = (int)matchedTracks.at(i_track)->hitPattern();

      // ------------------------------------------------------------------------------------------

      //float tmp_matchtrk_bend_chi2 = 0;

      tmp_matchtrk_dhits = 0;
      tmp_matchtrk_lhits = 0;

      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
          stubRefs = matchedTracks.at(i_track)->getStubRefs();
      int tmp_nstub = stubRefs.size();

      for (int is = 0; is < tmp_nstub; is++) {
        DetId detIdStub = theTrackerGeom->idToDet((stubRefs.at(is)->clusterRef(0))->getDetId())->geographicalId();
        /*
	MeasurementPoint coords = stubRefs.at(is)->clusterRef(0)->findAverageLocalCoordinatesCentered();
	const GeomDet* theGeomDet = theTrackerGeom->idToDet(detIdStub);
	Global3DPoint posStub = theGeomDet->surface().toGlobal( theGeomDet->topology().localPosition(coords) );
	*/

        int layer = -999999;
        if (detIdStub.subdetId() == StripSubdetector::TOB) {
          layer = static_cast<int>(tTopo->layer(detIdStub));
          tmp_matchtrk_lhits += pow(10, layer - 1);
        } else if (detIdStub.subdetId() == StripSubdetector::TID) {
          layer = static_cast<int>(tTopo->layer(detIdStub));
          tmp_matchtrk_dhits += pow(10, layer - 1);
        }

        // ------------------------------------------------------------------------------------------
      }
    }

    m_tp_pt->push_back(tmp_tp_pt);
    m_tp_eta->push_back(tmp_tp_eta);
    m_tp_phi->push_back(tmp_tp_phi);
    m_tp_dxy->push_back(tmp_tp_dxy);
    m_tp_z0->push_back(tmp_tp_z0);
    m_tp_d0->push_back(tmp_tp_d0);
    m_tp_z0_prod->push_back(tmp_tp_z0_prod);
    m_tp_d0_prod->push_back(tmp_tp_d0_prod);
    m_tp_pdgid->push_back(tmp_tp_pdgid);
    m_tp_nmatch->push_back(nMatch);
    m_tp_nstub->push_back(nStubTP);
    m_tp_eventid->push_back(tmp_eventid);
    m_tp_charge->push_back(tmp_tp_charge);

    m_matchtrk_pt->push_back(tmp_matchtrk_pt);
    m_matchtrk_eta->push_back(tmp_matchtrk_eta);
    m_matchtrk_phi->push_back(tmp_matchtrk_phi);
    m_matchtrk_z0->push_back(tmp_matchtrk_z0);
    m_matchtrk_d0->push_back(tmp_matchtrk_d0);
    m_matchtrk_chi2->push_back(tmp_matchtrk_chi2);
    m_matchtrk_chi2rphi->push_back(tmp_matchtrk_chi2rphi);
    m_matchtrk_chi2rz->push_back(tmp_matchtrk_chi2rz);
    m_matchtrk_bendchi2->push_back(tmp_matchtrk_bendchi2);
    m_matchtrk_MVA1->push_back(tmp_matchtrk_MVA1);
    m_matchtrk_nstub->push_back(tmp_matchtrk_nstub);
    m_matchtrk_dhits->push_back(tmp_matchtrk_dhits);
    m_matchtrk_lhits->push_back(tmp_matchtrk_lhits);
    m_matchtrk_seed->push_back(tmp_matchtrk_seed);
    m_matchtrk_hitpattern->push_back(tmp_matchtrk_hitpattern);

    // ----------------------------------------------------------------------------------------------
    // for tracking in jets
    // ----------------------------------------------------------------------------------------------

    if (TrackingInJets) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "check if TP/matched track is within jet";

      int tp_InJet = 0;
      int matchtrk_InJet = 0;
      int tp_InJetHighpt = 0;
      int matchtrk_InJetHighpt = 0;
      int tp_InJetVeryHighpt = 0;
      int matchtrk_InJetVeryHighpt = 0;

      for (int ij = 0; ij < (int)v_jets.size(); ij++) {
        float deta = tmp_tp_eta - (v_jets.at(ij)).eta();
        float dphi = tmp_tp_phi - (v_jets.at(ij)).phi();
        while (dphi > 3.14159)
          dphi = std::abs(2 * 3.14159 - dphi);
        float dR = sqrt(deta * deta + dphi * dphi);
        if (dR < 0.4) {
          tp_InJet = 1;
          if (v_jets_highpt.at(ij) == 1)
            tp_InJetHighpt = 1;
          if (v_jets_vhighpt.at(ij) == 1)
            tp_InJetVeryHighpt = 1;
          if (ij < NJETS)
            jets_tp_sumpt[ij] += tmp_tp_pt;
        }

        if (nMatch > 0) {
          deta = tmp_matchtrk_eta - (v_jets.at(ij)).eta();
          dphi = tmp_matchtrk_phi - (v_jets.at(ij)).phi();
          while (dphi > 3.14159)
            dphi = std::abs(2 * 3.14159 - dphi);
          dR = sqrt(deta * deta + dphi * dphi);
          if (dR < 0.4) {
            matchtrk_InJet = 1;
            if (v_jets_highpt.at(ij) == 1)
              matchtrk_InJetHighpt = 1;
            if (v_jets_vhighpt.at(ij) == 1)
              matchtrk_InJetVeryHighpt = 1;
            if (ij < NJETS)
              jets_matchtrk_sumpt[ij] += tmp_matchtrk_pt;
          }
        }
      }

      m_tp_injet->push_back(tp_InJet);
      m_tp_injet_highpt->push_back(tp_InJetHighpt);
      m_tp_injet_vhighpt->push_back(tp_InJetVeryHighpt);
      m_matchtrk_injet->push_back(matchtrk_InJet);
      m_matchtrk_injet_highpt->push_back(matchtrk_InJetHighpt);
      m_matchtrk_injet_vhighpt->push_back(matchtrk_InJetVeryHighpt);

    }  //end TrackingInJets

  }  //end loop tracking particles

  if (TrackingInJets) {
    for (int ij = 0; ij < (int)v_jets.size(); ij++) {
      if (ij < NJETS) {
        m_jet_eta->push_back((v_jets.at(ij)).eta());
        m_jet_phi->push_back((v_jets.at(ij)).phi());
        m_jet_pt->push_back((v_jets.at(ij)).pt());
        m_jet_tp_sumpt->push_back(jets_tp_sumpt[ij]);
        m_jet_trk_sumpt->push_back(jets_trk_sumpt[ij]);
        m_jet_matchtrk_sumpt->push_back(jets_matchtrk_sumpt[ij]);
      }
    }
  }

  eventTree->Fill();

}  // end of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackNtupleMaker);
