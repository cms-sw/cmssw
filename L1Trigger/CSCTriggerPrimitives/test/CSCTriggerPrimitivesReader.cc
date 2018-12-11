//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesReader
//
//   Description: Basic analyzer class which accesses ALCTs, CLCTs, and
//                correlated LCTs and plot various quantities.
//
//   Author List: S. Valuev, UCLA.
//
//
//   Modifications:
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSCTriggerPrimitivesReader.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

// MC data
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// MC tests
#include "L1Trigger/CSCTriggerPrimitives/test/CSCAnodeLCTAnalyzer.h"
#include "L1Trigger/CSCTriggerPrimitives/test/CSCCathodeLCTAnalyzer.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TText.h"
#include "TPaveLabel.h"
#include "TPostScript.h"
#include "TStyle.h"
#include "TROOT.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#define MAXSTUBS 100
using namespace std;

//-----------------
// Static variables
//-----------------


// Various useful constants
const double CSCTriggerPrimitivesReader::TWOPI = 2.*M_PI;
const string CSCTriggerPrimitivesReader::csc_type[CSC_TYPES] = {
  "ME1/1",  "ME1/2",  "ME1/3",  "ME1/A",  "ME2/1",  "ME2/2",
  "ME3/1",  "ME3/2",  "ME4/1",  "ME4/2"};
const string CSCTriggerPrimitivesReader::csc_type_plus[CSC_TYPES] = {
  "ME+1/1", "ME+1/2", "ME+1/3", "ME+1/A", "ME+2/1", "ME+2/2",
  "ME+3/1", "ME+3/2", "ME+4/1", "ME+4/2"};
const string CSCTriggerPrimitivesReader::csc_type_minus[CSC_TYPES] = {
  "ME-1/1", "ME-1/2", "ME-1/3", "ME-1/A", "ME-2/1", "ME-2/2",
  "ME-3/1", "ME-3/2", "ME-4/1", "ME-4/2"};
const int CSCTriggerPrimitivesReader::NCHAMBERS[CSC_TYPES] = {
  36, 36, 36, 36, 18, 36, 18, 36, 18, 36};
const int CSCTriggerPrimitivesReader::MAX_WG[CSC_TYPES] = {
  48,  64,  32,  48, 112,  64,  96,  64,  96,  64};//max. number of wiregroups
const int CSCTriggerPrimitivesReader::MAX_HS[CSC_TYPES] = {
  128, 160, 128,  96, 160, 160, 160, 160, 160, 160}; // max. # of halfstrips
const int CSCTriggerPrimitivesReader::ptype_TMB07[CSCConstants::NUM_CLCT_PATTERNS]= {
  -999,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0}; // "signed" pattern (== phiBend)

// LCT counters
int  CSCTriggerPrimitivesReader::numALCT   = 0;
int  CSCTriggerPrimitivesReader::numCLCT   = 0;
int  CSCTriggerPrimitivesReader::numLCTTMB = 0;
int  CSCTriggerPrimitivesReader::numLCTMPC = 0;

bool CSCTriggerPrimitivesReader::bookedHotWireHistos = false;
bool CSCTriggerPrimitivesReader::bookedALCTHistos    = false;
bool CSCTriggerPrimitivesReader::bookedCLCTHistos    = false;
bool CSCTriggerPrimitivesReader::bookedLCTTMBHistos  = false;
bool CSCTriggerPrimitivesReader::bookedLCTMPCHistos  = false;

bool CSCTriggerPrimitivesReader::bookedCompHistos    = false;

bool CSCTriggerPrimitivesReader::bookedResolHistos   = false;
bool CSCTriggerPrimitivesReader::bookedEfficHistos   = false;

bool CSCTriggerPrimitivesReader::printps = false;

// LUT for which ME1/1 wire group can cross which ME1/a halfstrip
// 1st index: WG number
// 2nd index: inclusive HS range
const int CSCTriggerPrimitivesReader::lut_wg_vs_hs_me1a[48][2] = {
  {0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
  {0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
  {0, 95},{0, 95},{0, 77},{0, 61},{0, 39},
  {0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1} };
// a modified LUT for ganged ME1a
const int CSCTriggerPrimitivesReader::lut_wg_vs_hs_me1ag[48][2] = {
  {0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
  {0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
  {0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
  {0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1} };

// LUT for which ME1/1 wire group can cross which ME1/b halfstrip
// 1st index: WG number
// 2nd index: inclusive HS range
const int CSCTriggerPrimitivesReader::lut_wg_vs_hs_me1b[48][2] = {
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
  {100, 127},{73, 127},{47, 127},{22, 127},{0, 127},
  {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
  {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
  {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
  {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
  {0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
  {0, 127},{0, 127},{0, 127},{0, 127},{0, 105},
  {0, 93},{0, 78},{0, 63} };

//init My ttree
void TreePerStub::init(int run, int event){

  t_EventNumberAnalyzed = -1;
  t_RUN = run;
  t_Event = event;
  t_nStubs = 0;
  t_nStubs_readout = 0;

  t_chamber = -1;
  t_ring = -1;
  t_station = -1;
  t_chambertype = -1;
  t_endcap = -2;
}

TTree *TreePerStub::bookTree(TTree *t, const std::string & name)
{
  edm::Service< TFileService > fs;
  t = fs->make<TTree>(name.c_str(), name.c_str());

  t->Branch("t_EventNumberAnalyzed",&t_EventNumberAnalyzed,"t_EventNumberAnalyzed/I");
  t->Branch("t_RUN",&t_RUN,"t_RUN/I");
  t->Branch("t_Event",&t_Event,"t_Event/I");
  t->Branch("t_nStubs",&t_nStubs,"t_nStubs/I");
  t->Branch("t_nStubs_readout",&t_nStubs_readout,"t_nStubs_readout/I");
  t->Branch("t_nStubs_ME119",&t_nStubs_ME119,"t_nStubs_ME119/I");
  t->Branch("t_nStubs_ME11no911",&t_nStubs_ME11no911,"t_nStubs_ME11no911/I");
  t->Branch("t_nStubs_noMEpm11",&t_nStubs_noMEpm11,"t_nStubs_noMEpm11/I");

  t->Branch("t_chamber", &t_chamber, "t_chamber/I");
  t->Branch("t_ring", &t_ring, "t_ring/I");
  t->Branch("t_endcap", &t_endcap, "t_endcap/I");
  t->Branch("t_station", &t_station, "t_station/I");
  t->Branch("t_chambertype", &t_chambertype, "t_chambertype/I");

  return t;
}


//init stub comparison struct
void MyStubComparison::init(int run, int event){

  nEvents = -1;
  nRUN = run;
  nEvent = event;
  firstfill =  false;
  totStubs_data = -1;
  totStubs_emul = -1;
  totStubs_emul_readout = -1;
  nStub_data = 0;
  nStub_emul = 0;
  chamber = -1;
  ring =-1;
  endcap =-1;
  station = -1;
  chambertype = -1;
  has_data = false;
  has_emul = false;

  npretrig = 0;
  quality_pretrig = -1;
  maxquality_pretrig = -1;
  bend_pretrig = -1;
  bx_pretrig = -1;
  key_hs_pretrig = -1;
  pattern_pretrig = -1;
  maxpattern_pretrig = -1;
  quality_data = -1;
  bend_data = -1;
  bx_data = -1;
  quality_emul = -1;
  bend_emul = -1;
  pattern_data = -1;
  pattern_emul = -1;
  bx_emul = -1;
  bx_corr_emul = -1;//corrected
  key_WG_data = -1;
  key_WG_emul = -1;
  key_hs_data  = -1;
  key_hs_emul = -1;
  WGcrossHS_data = false;
  WGcrossHS_emul = false;
  trknmb_data = -1;
  trknmb_emul = -1;
  dphi_data = -1;
  dphi_emul = -1;
  eta_data = -1;
  eta_emul = -1;
  phi_data = -1;
  phi_emul = -1;
}


TTree *MyStubComparison::bookTree(TTree *t, const std::string & name)
{
  edm::Service< TFileService > fs;
  t = fs->make<TTree>(name.c_str(), name.c_str());

  t->Branch("nEvents",&nEvents);
  t->Branch("nRUN",&nRUN);
  t->Branch("nEvent",&nEvent);
  t->Branch("firstfill",&firstfill);
  t->Branch("totStubs_data",&totStubs_data);
  t->Branch("totStubs_emul",&totStubs_emul);
  t->Branch("totStubs_emul_readout",&totStubs_emul_readout);
  t->Branch("nStub_data",&nStub_data);
  t->Branch("nStub_emul",&nStub_emul);

  t->Branch("chamber",&chamber);
  t->Branch("ring",&ring);
  t->Branch("endcap",&endcap);
  t->Branch("station",&station);
  t->Branch("chambertype",&chambertype);
  t->Branch("has_data",&has_data);
  t->Branch("has_emul",&has_emul);
  t->Branch("quality_data",&quality_data);
  t->Branch("quality_emul",&quality_emul);
  t->Branch("npretrig",&npretrig);
  t->Branch("quality_pretrig",&quality_pretrig);
  t->Branch("maxquality_pretrig",&maxquality_pretrig);
  t->Branch("pattern_data",&pattern_data);
  t->Branch("pattern_emul",&pattern_emul);
  t->Branch("pattern_pretrig",&pattern_pretrig);
  t->Branch("maxpattern_pretrig",&maxpattern_pretrig);
  t->Branch("bend_data",&bend_data);
  t->Branch("bx_data",&bx_data);
  t->Branch("fullbx_data",&fullbx_data);
  t->Branch("bend_emul",&bend_emul);
  t->Branch("bx_emul",&bx_emul);
  t->Branch("fullbx_emul",&fullbx_emul);
  t->Branch("bend_pretrig",&bend_pretrig);
  t->Branch("bx_pretrig",&bx_pretrig);
  t->Branch("bx_corr_emul",&bx_corr_emul);
  t->Branch("WGcrossHS_data",&WGcrossHS_data);
  t->Branch("WGcrossHS_emul",&WGcrossHS_emul);
  t->Branch("key_WG_data",&key_WG_data);
  t->Branch("key_WG_emul",&key_WG_emul);
  t->Branch("key_hs_data",&key_hs_data);
  t->Branch("key_hs_emul",&key_hs_emul);
  t->Branch("key_hs_pretrig",&key_hs_pretrig);
  t->Branch("trknmb_data",&trknmb_data);
  t->Branch("trknmb_emul",&trknmb_emul);
  t->Branch("dphi_data",&dphi_data);
  t->Branch("dphi_emul",&dphi_emul);
  t->Branch("eta_data",&eta_data);
  t->Branch("eta_emul",&eta_emul);
  t->Branch("phi_data", &phi_data);
  t->Branch("phi_emul", &phi_emul);

  return t;

}


//----------------
// Constructor  --
//----------------
CSCTriggerPrimitivesReader::CSCTriggerPrimitivesReader(const edm::ParameterSet& conf) : eventsAnalyzed(0) {
  edm::Service<TFileService> fs;
  //  rootFileName = conf.getUntrackedParameter<string>("rootFileName","TPEHists.root");
  // Create the root file for the histograms
  //  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  //  theFile->cd();

  // Various input parameters.

  printps = conf.getParameter<bool>("printps");
  dataLctsIn_ = conf.getParameter<bool>("dataLctsIn");
  emulLctsIn_ = conf.getParameter<bool>("emulLctsIn");
  edm::ParameterSet commonParams =
    conf.getParameter<edm::ParameterSet>("commonParam");

  // Switch for a new (2007) version of the TMB firmware.
  gangedME1a = commonParams.getParameter<bool>("gangedME1a");

  // is it (non-upgrade algorithm) run along with upgrade one?
  plotME1A = true;
  plotME42 = true;
  lctProducerData_ = conf.getUntrackedParameter<string>("CSCLCTProducerData",
                                                        "cscunpacker");
  mpclctProducerData_ = conf.getUntrackedParameter<string>("CSCMPCLCTProducerData",
                                                           "csctfDigis");
  lctProducerEmul_ = conf.getUntrackedParameter<string>("CSCLCTProducerEmul",
                                                        "cscTriggerPrimitiveDigis");

  simHitProducer_ = conf.getParameter<edm::InputTag>("CSCSimHitProducer");
  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");

  simHit_token_     = consumes<edm::PSimHitContainer>(simHitProducer_);
  wireDigi_token_   = consumes<CSCWireDigiCollection>(wireDigiProducer_);
  compDigi_token_   = consumes<CSCComparatorDigiCollection>(compDigiProducer_);

  alcts_d_token_    = consumes<CSCALCTDigiCollection>(edm::InputTag(lctProducerData_, "MuonCSCALCTDigi"));
  clcts_d_token_    = consumes<CSCCLCTDigiCollection>(edm::InputTag(lctProducerData_, "MuonCSCCLCTDigi"));
  lcts_tmb_d_token_ = consumes<CSCCorrelatedLCTDigiCollection>(edm::InputTag(lctProducerData_, "MuonCSCCorrelatedLCTDigi"));
  lcts_mpc_d_token_ = consumes<CSCCorrelatedLCTDigiCollection>(edm::InputTag(mpclctProducerData_));

  alcts_e_token_    = consumes<CSCALCTDigiCollection>(edm::InputTag(lctProducerEmul_));
  clcts_e_token_    = consumes<CSCCLCTDigiCollection>(edm::InputTag(lctProducerEmul_));
  pretrigs_e_token_ = consumes<CSCCLCTPreTriggerDigiCollection>(edm::InputTag(lctProducerEmul_));
  lcts_tmb_e_token_ = consumes<CSCCorrelatedLCTDigiCollection>(edm::InputTag(lctProducerEmul_));
  lcts_mpc_e_token_ = consumes<CSCCorrelatedLCTDigiCollection>(edm::InputTag(lctProducerEmul_, "MPCSORTED"));

  consumesMany<edm::HepMCProduct>();
  resultsFileNamesPrefix_ = conf.getUntrackedParameter<string>("resultsFileNamesPrefix","");
  checkBadChambers_ = conf.getUntrackedParameter<bool>("checkBadChambers",true);
  debug = conf.getUntrackedParameter<bool>("debug", false);
  dataIsAnotherMC_ = conf.getUntrackedParameter<bool>("dataIsAnotherMC", false);

  //rootFileName = conf.getUntrackedParameter<string>("rootFileName");
  // Create the root file.
  // Not sure we really need it - comment out for now. -Slava.
  //theFile = new TFile(rootFileName.c_str(), "RECREATE");
  //theFile->cd();
  stub_tree[0] = stubs_comparison[0].bookTree(stub_tree[0],"alcttree");
  stub_tree[1] = stubs_comparison[1].bookTree(stub_tree[1],"clcttree");
  stub_tree[2] = stubs_comparison[2].bookTree(stub_tree[2],"lcttree");
  stub_tree[3] = stubs_comparison[3].bookTree(stub_tree[3],"mpclcttree");
  // Per event TTree
  event_tree[0] = perStub[0].bookTree(event_tree[0],"Ev_alcttree");
  event_tree[1] = perStub[1].bookTree(event_tree[1],"Ev_emul_alcttree");
  event_tree[2] = perStub[2].bookTree(event_tree[2],"Ev_clcttree");
  event_tree[3] = perStub[3].bookTree(event_tree[3],"Ev_emul_clcttree");
  event_tree[4] = perStub[4].bookTree(event_tree[4],"Ev_lcttree");
  event_tree[5] = perStub[5].bookTree(event_tree[5],"Ev_emul_lcttree");
  // My favourite ROOT settings.
  setRootStyle();
}

//----------------
// Destructor   --
//----------------
CSCTriggerPrimitivesReader::~CSCTriggerPrimitivesReader() {
  //  histos->writeHists(theFile);
  //  theFile->Close();
  //delete theFile;
}


int CSCTriggerPrimitivesReader::maxRing(int station)
{
  if (station == 1) {
    if (plotME1A) return 4;
    else  return 3;
  }
  return 2;
}


void CSCTriggerPrimitivesReader::analyze(const edm::Event& ev,
                                         const edm::EventSetup& setup) {
  ++eventsAnalyzed;
  //if (ev.id().event()%10 == 0)
  LogTrace("CSCTriggerPrimitivesReader")
    << "\n** CSCTriggerPrimitivesReader: processing run #"
    << ev.id().run() << " event #" << ev.id().event()
    << "; events so far: " << eventsAnalyzed << " **";
  RUN_ = ev.id().run();
  Event_ = ev.id().event();

  // Find the geometry for this event & cache it.  Needed in LCTAnalyzer
  // modules.
  edm::ESHandle<CSCGeometry> cscGeom;
  setup.get<MuonGeometryRecord>().get(cscGeom);
  geom_ = &*cscGeom;

  // Find conditions data for bad chambers & cache it.  Needed for efficiency
  // calculations.
  if (checkBadChambers_) {
    edm::ESHandle<CSCBadChambers> pBad;
    setup.get<CSCBadChambersRcd>().get(pBad);
    badChambers_ = pBad.product();
  }

  // Get the collections of ALCTs, CLCTs, and correlated LCTs from event.
  edm::Handle<CSCALCTDigiCollection> alcts_data;
  edm::Handle<CSCCLCTDigiCollection> clcts_data;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_tmb_data;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_mpc_data;
  edm::Handle<CSCALCTDigiCollection> alcts_emul;
  edm::Handle<CSCCLCTDigiCollection> clcts_emul;
  edm::Handle<CSCCLCTPreTriggerDigiCollection> pretrigs_emul;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_tmb_emul;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_mpc_emul;

  // Data
  if (dataLctsIn_) {
    HotWires(ev);
    //    ev.getByLabel(lctProducerData_,  alcts_data);
    //    ev.getByLabel(lctProducerData_,  clcts_data);
    //    ev.getByLabel(lctProducerData_,  lcts_tmb_data);
    ev.getByToken(alcts_d_token_, alcts_data);
    ev.getByToken(clcts_d_token_, clcts_data);
    ev.getByToken(lcts_tmb_d_token_, lcts_tmb_data);
    ev.getByToken(lcts_mpc_d_token_, lcts_mpc_data);

    if (!alcts_data.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of ALCTs with label MuonCSCALCTDigi"
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }
    if (!clcts_data.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of CLCTs with label MuonCSCCLCTDigi"
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }
    if (!lcts_tmb_data.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of correlated LCTs with label"
        << " MuonCSCCorrelatedLCTDigi requested, but not found in the"
        << " event... Skipping the rest +++\n";
      return;
    }
    /*
      if (!lcts_mpc_data.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
      << "+++ Warning: Collection of MPC correlated LCTs with label"
      << " MuonCSCCorrelatedLCTDigi + MCPSorted requested, but not found in the"
      << " event... Skipping the rest +++\n";
      //return;
      }*/
  }

  // Emulator
  if (emulLctsIn_) {
    //    ev.getByLabel(lctProducerEmul_,              alcts_emul);
    //    ev.getByLabel(lctProducerEmul_,              clcts_emul);
    //    ev.getByLabel(lctProducerEmul_,              lcts_tmb_emul);
    //    ev.getByLabel(lctProducerEmul_, "MPCSORTED", lcts_mpc_emul);
    ev.getByToken(alcts_e_token_, alcts_emul);
    ev.getByToken(clcts_e_token_, clcts_emul);
    ev.getByToken(pretrigs_e_token_, pretrigs_emul);
    ev.getByToken(lcts_tmb_e_token_, lcts_tmb_emul);
    ev.getByToken(lcts_mpc_e_token_, lcts_mpc_emul);

    if (!alcts_emul.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of emulated ALCTs"
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }
    if (!clcts_emul.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of emulated CLCTs"
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }
    if (!lcts_tmb_emul.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of emulated correlated LCTs"
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }
    /*
      if (!lcts_mpc_emul.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
      << "+++ Warning: Collection of emulated correlated LCTs (MPCs)"
      << " requested, but not found in the event... Skipping the rest +++\n";
      //return;
      }*/
  }

  // Fill histograms with reconstructed or emulated quantities.  If both are
  // present, plot LCTs in data.
  if (dataLctsIn_) {
    fillALCTHistos(alcts_data.product());
    fillCLCTHistos(clcts_data.product());
    fillLCTTMBHistos(lcts_tmb_data.product());
  }
  else if (emulLctsIn_) {
    fillALCTHistos(alcts_emul.product());
    fillCLCTHistos(clcts_emul.product());
    fillLCTTMBHistos(lcts_tmb_emul.product());
    //fillLCTMPCHistos(lcts_mpc_emul.product());
  }

  // Compare LCTs in the data with the ones produced by the emulator.
  if (dataLctsIn_ && emulLctsIn_) {
    compare(alcts_data.product(), alcts_emul.product(),
            clcts_data.product(), clcts_emul.product(),
            pretrigs_emul.product(), lcts_tmb_data.product(), lcts_tmb_emul.product()
            );
  }
  // Fill MC-based resolution/efficiency histograms, if needed.
  if (emulLctsIn_) {
    MCStudies(ev, alcts_emul.product(), clcts_emul.product());
  }
} // analyze

void CSCTriggerPrimitivesReader::endJob() {
  // Note: all operations involving ROOT should be placed here and not in the
  // destructor.
  // Plot histos if they were booked/filled.
  if(printps) {
    if (bookedALCTHistos)   drawALCTHistos();
    if (bookedCLCTHistos)   drawCLCTHistos();
    if (bookedLCTTMBHistos) drawLCTTMBHistos();
    //if (bookedLCTMPCHistos) drawLCTMPCHistos();

    if (bookedCompHistos)   drawCompHistos();

    if (bookedResolHistos)  drawResolHistos();
    if (bookedEfficHistos)  drawEfficHistos();
  }
  //drawHistosForTalks();

  //theFile->cd();
  //theFile->Write();
  //theFile->Close();

  // Job summary.
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "\n  Average number of ALCTs/event = "
    << static_cast<float>(numALCT)/eventsAnalyzed << endl;
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "  Average number of CLCTs/event = "
    << static_cast<float>(numCLCT)/eventsAnalyzed << endl;
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "  Average number of TMB LCTs/event = "
    << static_cast<float>(numLCTTMB)/eventsAnalyzed << endl;
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "  Average number of MPC LCTs/event = "
    << static_cast<float>(numLCTMPC)/eventsAnalyzed << endl;

  if (bookedEfficHistos) {
    {
      edm::LogInfo("CSCTriggerPrimitivesReader") << "\n  ALCT efficiencies:";
      double tot_simh = 0.0, tot_alct = 0.0;
      for (int idh = 0; idh < CSC_TYPES; idh++) {
        double simh = hEfficHitsEtaCsc[idh]->Integral();
        double alct = hEfficALCTEtaCsc[idh]->Integral();
        double eff  = 0.;
        if (simh > 0) eff = alct/simh;
        edm::LogInfo("CSCTriggerPrimitivesReader")
          << "    " << csc_type[idh]
          << ": alct = " << alct << ", simh = " << simh << " eff = " << eff;
        tot_simh += simh;
        tot_alct += alct;
      }
      edm::LogInfo("CSCTriggerPrimitivesReader")
        << "    overall: alct = " << tot_alct << ", simh = " << tot_simh
        << " eff = " << tot_alct/tot_simh;
    }

    {
      edm::LogInfo("CSCTriggerPrimitivesReader") << "\n  CLCT efficiencies:";
      double tot_simh = 0.0, tot_clct = 0.0;
      for (int idh = 0; idh < CSC_TYPES; idh++) {
        double simh = hEfficHitsEtaCsc[idh]->Integral();
        double clct = hEfficCLCTEtaCsc[idh]->Integral();
        double eff  = 0.;
        if (simh > 0.) eff = clct/simh;
        edm::LogInfo("CSCTriggerPrimitivesReader")
          << "    " << csc_type[idh]
          << ": clct = " << clct << ", simh = " << simh << " eff = " << eff;
        tot_simh += simh;
        tot_clct += clct;
      }
      edm::LogInfo("CSCTriggerPrimitivesReader")
        << "    overall: clct = " << tot_clct << ", simh = " << tot_simh
        << " eff = " << tot_clct/tot_simh;
    }
  }

  if (bookedResolHistos) {
    double cor = 0.0, tot = 0.0;
    cor = hResolDeltaHS->GetBinContent(hResolDeltaHS->FindBin(0.));
    tot = hResolDeltaHS->GetEntries();
    edm::LogInfo("CSCTriggerPrimitivesReader")
      << "\n  Correct half-strip assigned in " << cor << "/" << tot
      << " = " << cor/tot << " of half-strip CLCTs";
    //cor = hResolDeltaDS->GetBinContent(hResolDeltaDS->FindBin(0.));
    //tot = hResolDeltaDS->GetEntries();
    //edm::LogInfo("CSCTriggerPrimitivesReader")
    //  << "  Correct di-strip assigned in " << cor << "/" << tot
    //  << " = " << cor/tot << " of di-strip CLCTs";
    cor = hResolDeltaWG->GetBinContent(hResolDeltaWG->FindBin(0.));
    tot = hResolDeltaWG->GetEntries();
    edm::LogInfo("CSCTriggerPrimitivesReader")
      << "  Correct wire group assigned in " << cor << "/" << tot
      << " = " << cor/tot << " of ALCTs";
  }
}

//---------------
// ROOT settings
//---------------
void CSCTriggerPrimitivesReader::setRootStyle() {
  TH1::AddDirectory(false);

  gROOT->SetStyle("Plain");
  gStyle->SetFillColor(0);
  gStyle->SetOptDate();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(1111);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetMarkerSize(0.5);
  gStyle->SetMarkerStyle(8);
  gStyle->SetGridStyle(3);
  gStyle->SetPaperSize(TStyle::kA4);
  gStyle->SetStatW(0.25); // width of statistics box; default is 0.19
  gStyle->SetStatH(0.10); // height of statistics box; default is 0.1
  gStyle->SetStatFormat("6.4g");  // leave default format for now
  gStyle->SetTitleSize(0.055, "");   // size for pad title; default is 0.02
  // Really big; useful for talks.
  //gStyle->SetTitleSize(0.1, "");   // size for pad title; default is 0.02
  gStyle->SetLabelSize(0.05, "XYZ"); // size for axis labels; default is 0.04
  gStyle->SetStatFontSize(0.06);     // size for stat. box
  gStyle->SetTitleFont(32, "XYZ"); // times-bold-italic font (p. 153) for axes
  gStyle->SetTitleFont(32, "");    // same for pad title
  gStyle->SetLabelFont(32, "XYZ"); // same for axis labels
  gStyle->SetStatFont(32);         // same for stat. box
  gStyle->SetLabelOffset(0.006, "Y"); // default is 0.005
}

//---------------------
// Histograms for LCTs
//---------------------
void CSCTriggerPrimitivesReader::bookHotWireHistos() {
  edm::Service<TFileService> fs;
  hHotWire1  = fs->make<TH1F>("hHotWire1", "hHotWire1",570*6*112,0,570*6*112);
  hHotCham1  = fs->make<TH1F>("hHotCham1", "hHotCham1",570,0,570);
  bookedHotWireHistos = true;
}

void CSCTriggerPrimitivesReader::bookALCTHistos() {
  string s;

  edm::Service<TFileService> fs;
  hAlctPerEvent  = fs->make<TH1F>("ALCTs_per_event", "ALCTs per event",     31, -0.5,  30.5);
  hAlctPerChamber= fs->make<TH1F>("ALCTs_per_chamber", "ALCTs per chamber",    4, -0.5,   3.5);
  hAlctPerCSC    = fs->make<TH1F>("ALCTs_per_CSCtype", "ALCTs per CSC type",  10, -0.5,   9.5);
  for (int i = 0; i < MAX_ENDCAPS; i++) { // endcaps
    for (int j = 0; j < CSC_TYPES; j++) { // station/ring
      float csc_max = NCHAMBERS[j] + 0.5;
      char asdf[256];
      sprintf(asdf,"ALCTs_%i",i*CSC_TYPES+j);
      if (i == 0) s = "ALCTs, " + csc_type_plus[j];
      else        s = "ALCTs, " + csc_type_minus[j];
      hAlctCsc[i][j] = fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
    }
  }

  hAlctValid     = fs->make<TH1F>("ALCT_validity", "ALCT validity",        3, -0.5,   2.5);
  hAlctQuality   = fs->make<TH1F>("ALCT_quality", "ALCT quality",         5, -0.5,   4.5);
  hAlctAccel     = fs->make<TH1F>("ALCT_accel_flag", "ALCT accel. flag",     3, -0.5,   2.5);
  hAlctCollis    = fs->make<TH1F>("ALCT_collision_flag", "ALCT collision. flag", 3, -0.5,   2.5);
  hAlctKeyGroup  = fs->make<TH1F>("ALCT_key_wiregroup", "ALCT key wiregroup", 120, -0.5, 119.5);
  hAlctBXN       = fs->make<TH1F>("ALCT_bx", "ALCT bx",             20, -0.5,  19.5);

  hAlctKeyGroupME11 = fs->make<TH1F>("hAlctKeyGroupME11", "ALCT key wiregroup ME1/1", 50, -0.5, 49.5);

  bookedALCTHistos = true;
  std::cout <<" bookedALCTHistos " << std::endl;
}

void CSCTriggerPrimitivesReader::bookCLCTHistos() {
  string s;

  edm::Service<TFileService> fs;
  hClctPerEvent  = fs->make<TH1F>("CLCTs_per_event", "CLCTs per event",    31, -0.5, 30.5);
  hClctPerChamber= fs->make<TH1F>("CLCTs_per_chamber", "CLCTs per chamber",   3, -0.5,  2.5);
  hClctPerCSC    = fs->make<TH1F>("CLCTs_per_CSCtype", "CLCTs per CSC type", 10, -0.5,  9.5);
  for (int i = 0; i < MAX_ENDCAPS; i++) { // endcaps
    for (int j = 0; j < CSC_TYPES; j++) { // station/ring
      char asdf[256];
      sprintf(asdf,"CLCTs_%i",i*CSC_TYPES+j);
      float csc_max = NCHAMBERS[j] + 0.5;
      if (i == 0) s = "CLCTs, " + csc_type_plus[j];
      else        s = "CLCTs, " + csc_type_minus[j];
      hClctCsc[i][j] = fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
    }
  }

  hClctValid     = fs->make<TH1F>("CLCT_validity", "CLCT validity",       3, -0.5,  2.5);
  hClctQuality   = fs->make<TH1F>("CLCT_layers_hit", "CLCT layers hit",     9, -0.5,  8.5);
  hClctStripType = fs->make<TH1F>("CLCT_strip_type", "CLCT strip type",     3, -0.5,  2.5);
  hClctSign      = fs->make<TH1F>("CLCT_sing_(L/R)", "CLCT sign (L/R)",     3, -0.5,  2.5);
  hClctCFEB      = fs->make<TH1F>("CLCT_cfeb_#", "CLCT cfeb #",         6, -0.5,  5.5);
  hClctBXN       = fs->make<TH1F>("CLCT_bx", "CLCT bx",            20, -0.5, 19.5);

  hClctKeyStrip[0] = fs->make<TH1F>("CLCT_keystrip_distrips","CLCT keystrip, distrips",   40, -0.5,  39.5);
  //hClctKeyStrip[0] = fs->make<TH1F>("","CLCT keystrip, distrips",  160, -0.5, 159.5);
  hClctKeyStrip[1] = fs->make<TH1F>("CLCT_keystrip_halfstrips","CLCT keystrip, halfstrips",160, -0.5, 159.5);
  hClctPattern[0]  = fs->make<TH1F>("CLCT_pattern_distrips","CLCT pattern, distrips",    13, -0.5,  12.5);
  hClctPattern[1]  = fs->make<TH1F>("CLCT_pattern_halfstrips","CLCT pattern, halfstrips",  13, -0.5,  12.5);

  for (int i = 0; i < CSC_TYPES; i++) {
    char asdf[256];
    string s1 = "CLCT bend, " + csc_type[i];
    sprintf(asdf,"CLCT_bend0_%i",i+1);
    hClctBendCsc[i][0] = fs->make<TH1F>(asdf, s1.c_str(),  5, -0.5, 4.5);
    sprintf(asdf,"CLCT_bend1_%i",i+1);
    hClctBendCsc[i][1] = fs->make<TH1F>(asdf, s1.c_str(),  5, -0.5, 4.5);

    sprintf(asdf,"CLCT_keystrip_%i",i+1);
    string s2 = "CLCT keystrip, " + csc_type[i];
    int max_ds = MAX_HS[i]/4;
    hClctKeyStripCsc[i]   = fs->make<TH1F>(asdf, s2.c_str(), max_ds, 0., max_ds);
  }

  hClctKeyStripME11 = fs->make<TH1F>("hClctKeyStripME11","CLCT keystrip, halfstrips ME1/1",161, -0.5, 160.5);

  bookedCLCTHistos = true;
  std::cout <<" bookedCLCTHistos " << std::endl;
}

void CSCTriggerPrimitivesReader::bookLCTTMBHistos() {
  string s;

  edm::Service<TFileService> fs;
  hLctTMBPerEvent  = fs->make<TH1F>("LCTs_per_event", "LCTs per event",    31, -0.5, 30.5);
  hLctTMBPerChamber= fs->make<TH1F>("LCTs_per_chamber", "LCTs per chamber",   3, -0.5,  2.5);
  hLctTMBPerCSC    = fs->make<TH1F>("LCTs_per_CSCtype", "LCTs per CSC type", 10, -0.5,  9.5);
  hCorrLctTMBPerCSC= fs->make<TH1F>("CorrLCTs_per_CSCtype", "Corr. LCTs per CSC type", 10, -0.5, 9.5);
  hLctTMBEndcap    = fs->make<TH1F>("LCTS_endcap", "Endcap",             4, -0.5,  3.5);
  hLctTMBStation   = fs->make<TH1F>("LCTS_station", "Station",            6, -0.5,  5.5);
  hLctTMBSector    = fs->make<TH1F>("LCTS_sector", "Sector",             8, -0.5,  7.5);
  hLctTMBRing      = fs->make<TH1F>("LCTS_ring", "Ring",               5, -0.5,  4.5);
  for (int i = 0; i < MAX_ENDCAPS; i++) { // endcaps
    for (int j = 0; j < CSC_TYPES; j++) { // station/ring
      char asdf[256];
      float csc_max = NCHAMBERS[j] + 0.5;
      if (i == 0) s = "LCTs, " + csc_type_plus[j];
      else        s = "LCTs, " + csc_type_minus[j];
      sprintf(asdf,"LCTs_%i",i*CSC_TYPES+j);
      hLctTMBCsc[i][j] = fs->make<TH1F>(s.c_str(), s.c_str(), NCHAMBERS[j], 0.5, csc_max);
    }
  }

  hLctTMBValid     = fs->make<TH1F>("LCT_validity", "LCT validity",        3, -0.5,   2.5);
  hLctTMBQuality   = fs->make<TH1F>("LCT_quality", "LCT quality",        17, -0.5,  16.5);
  hLctTMBKeyGroup  = fs->make<TH1F>("LCT_key_wiregroup", "LCT key wiregroup", 120, -0.5, 119.5);
  hLctTMBKeyStrip  = fs->make<TH1F>("LCT_key_strip", "LCT key strip",     160, -0.5, 159.5);
  hLctTMBStripType = fs->make<TH1F>("LCT_strip_type", "LCT strip type",      3, -0.5,   2.5);
  hLctTMBPattern   = fs->make<TH1F>("LCT_pattern", "LCT pattern",        13, -0.5,  12.5);
  hLctTMBBend      = fs->make<TH1F>("LCT_bend", "LCT L/R bend",        3, -0.5,   2.5);
  hLctTMBBXN       = fs->make<TH1F>("LCT_bx", "LCT bx",             20, -0.5,  19.5);

  // LCT quantities per station
  char histname[60];
  for (int istat = 0; istat < MAX_STATIONS; istat++) {
    sprintf(histname, "LCT_CSCId, station %d", istat+1);
    hLctTMBChamber[istat] = fs->make<TH1F>("", histname,  10, -0.5, 9.5);
  }

  hLctTMBKeyGroupME11  = fs->make<TH1F>("hLctTMBKeyGroupME11", "LCT key wiregroup ME1/1", 50, -0.5, 49.5);
  hLctTMBKeyStripME11  = fs->make<TH1F>("hLctTMBKeyStripME11", "LCT key strip ME1/1",	  161, -0.5, 160.5);

  bookedLCTTMBHistos = true;
  std::cout <<" bookedLCTTMBHistos " << std::endl;
}

int CSCTriggerPrimitivesReader::chamberIXi(CSCDetId id) {
  //    1/1 1/2 1/3 2/1 2/2 3/1 3/2 4/1 4/2 -1/1 -1/2 -1/3 -2/1 -2/2 -3/1 -3/2 -4/1 -4/2
  //ix= 0   1   2   3   4   5   6   7   8    9    10   11   12   13   14   15   16   17
  int ix=0;
  if(id.station()!=1) {

    ix=(id.station()-2)*2+3;
  }
  ix+=id.ring()-1;
  if(id.endcap()==2) {
    ix+=9;
  }
  return ix;
}

int CSCTriggerPrimitivesReader::chamberIX(CSCDetId id) {
  int ix=1;
  if(id.station()!=1) {
    ix=(id.station()-2)*2+3+1;
  }
  ix+=id.ring()-1;
  if(id.endcap()==2) {
    ix*=-1;
  }
  return ix;
}

int CSCTriggerPrimitivesReader::chamberSerial( CSCDetId id ) {
  int st = id.station();
  int ri = id.ring();
  int ch = id.chamber();
  int ec = id.endcap();
  int kSerial = ch;
  if (st == 1 && ri == 1) kSerial = ch;
  if (st == 1 && ri == 2) kSerial = ch + 36;
  if (st == 1 && ri == 3) kSerial = ch + 72;
  if (st == 1 && ri == 4) kSerial = ch;
  if (st == 2 && ri == 1) kSerial = ch + 108;
  if (st == 2 && ri == 2) kSerial = ch + 126;
  if (st == 3 && ri == 1) kSerial = ch + 162;
  if (st == 3 && ri == 2) kSerial = ch + 180;
  if (st == 4 && ri == 1) kSerial = ch + 216;
  if (st == 4 && ri == 2) kSerial = ch + 234;  // one day...
  if (ec == 2) kSerial = kSerial + 300;
  return kSerial;
}


void CSCTriggerPrimitivesReader::bookLCTMPCHistos() {
  edm::Service<TFileService> fs;
  hLctMPCPerEvent  = fs->make<TH1F>("MPC_per_event", "LCTs per event",    31, -0.5, 30.5);
  hLctMPCPerCSC    = fs->make<TH1F>("MPC_per_CSCtype", "LCTs per CSC type", 10, -0.5,  9.5);
  hCorrLctMPCPerCSC= fs->make<TH1F>("CorrMPC_per_CSCtype", "Corr. LCTs per CSC type", 10, -0.5,9.5);
  hLctMPCEndcap    = fs->make<TH1F>("MPC_Endcap", "Endcap",             4, -0.5,  3.5);
  hLctMPCStation   = fs->make<TH1F>("MPC_Station", "Station",            6, -0.5,  5.5);
  hLctMPCSector    = fs->make<TH1F>("MPC_Sector", "Sector",             8, -0.5,  7.5);
  hLctMPCRing      = fs->make<TH1F>("MPC_Ring", "Ring",               5, -0.5,  4.5);

  hLctMPCValid     = fs->make<TH1F>("MPC_validity", "LCT validity",        3, -0.5,   2.5);
  hLctMPCQuality   = fs->make<TH1F>("MPC_quality", "LCT quality",        17, -0.5,  16.5);
  hLctMPCKeyGroup  = fs->make<TH1F>("MPC_key_wiregroup", "LCT key wiregroup", 120, -0.5, 119.5);
  hLctMPCKeyStrip  = fs->make<TH1F>("MPC_key_strip", "LCT key strip",     160, -0.5, 159.5);
  hLctMPCStripType = fs->make<TH1F>("MPC_strip_type", "LCT strip type",      3, -0.5,   2.5);
  hLctMPCPattern   = fs->make<TH1F>("MPC_pattern", "LCT pattern",        13, -0.5,  12.5);
  hLctMPCBend      = fs->make<TH1F>("MPC_bend", "LCT L/R bend",        3, -0.5,   2.5);
  hLctMPCBXN       = fs->make<TH1F>("MPC_bx", "LCT bx",             20, -0.5,  19.5);

  // LCT quantities per station
  char histname[60];
  for (int istat = 0; istat < MAX_STATIONS; istat++) {
    sprintf(histname, "MPC_CSCId, station %d", istat+1);
    hLctMPCChamber[istat] = fs->make<TH1F>("", histname,  10, -0.5, 9.5);
  }

  hLctMPCKeyGroupME11  = fs->make<TH1F>("hLctMPCKeyGroupME11", "MPC LCT key wiregroup ME1/1", 50, -0.5, 49.5);
  hLctMPCKeyStripME11  = fs->make<TH1F>("hLctMPCKeyStripME11", "MPC LCT key strip ME1/1",     161, -0.5, 160.5);

  bookedLCTMPCHistos = true;
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               <<"bookedLCTMPCHistos  ";
  std::cout <<" bookedLCTMPCHistos " << std::endl;

}

void CSCTriggerPrimitivesReader::bookCompHistos() {
  string s;



  edm::Service<TFileService> fs;
  //  hAlctCompMatch;

  hAlctCompFound = fs->make<TH1F>("h_ALCT_found","h_ALCT_found",600,0.5,600.5);
  hAlctCompFound2 = fs->make<TH2F>("h_ALCT_found2","h_ALCT_found2",19,-9.5,9.5,36,0.5,36.5);
  hAlctCompFound2x = fs->make<TH2F>("h_ALCT_found2x","h_ALCT_found2x",19,-9.5,9.5,18,0.5,36.5);


  hAlctCompSameN = fs->make<TH1F>("h_ALCT_SameN","h_ALCT_SameN",600,0.5,600.5);
  hAlctCompSameN2 = fs->make<TH2F>("h_ALCT_SameN2","h_ALCT_SameN2",19,-9.5,9.5,36,0.5,36.5);
  hAlctCompSameN2x = fs->make<TH2F>("h_ALCT_SameN2x","h_ALCT_SameN2x",19,-9.5,9.5,18,0.5,36.5);


  hAlctCompMatch = fs->make<TH1F>("h_ALCT_match","h_ALCT_match",600,0.5,600.5);
  hAlctCompMatch2 = fs->make<TH2F>("h_ALCT_match2","h_ALCT_match2",19,-9.5,9.5,36,0.5,36.5);
  hAlctCompMatch2x = fs->make<TH2F>("h_ALCT_match2x","h_ALCT_match2x",19,-9.5,9.5,18,0.5,36.5);

  hAlctCompTotal = fs->make<TH1F>("h_ALCT_total","h_ALCT_total",600,0.5,600.5);
  hAlctCompTotal2 = fs->make<TH2F>("h_ALCT_total2","h_ALCT_total2",19,-9.5,9.5,36,0.5,36.5);
  hAlctCompTotal2x = fs->make<TH2F>("h_ALCT_total2x","h_ALCT_total2x",19,-9.5,9.5,18,0.5,36.5);

  hClctCompFound = fs->make<TH1F>("h_CLCT_found","h_CLCT_found",600,0.5,600.5);
  hClctCompFound2 = fs->make<TH2F>("h_CLCT_found2","h_CLCT_found2",19,-9.5,9.5,36,0.5,36.5);
  hClctCompFound2x = fs->make<TH2F>("h_CLCT_found2x","h_CLCT_found2x",19,-9.5,9.5,18,0.5,36.5);

  hClctCompSameN = fs->make<TH1F>("h_CLCT_SameN","h_CLCT_SameN",600,0.5,600.5);
  hClctCompSameN2 = fs->make<TH2F>("h_CLCT_SameN2","h_CLCT_SameN2",19,-9.5,9.5,36,0.5,36.5);
  hClctCompSameN2x = fs->make<TH2F>("h_CLCT_SameN2x","h_CLCT_SameN2x",19,-9.5,9.5,18,0.5,36.5);

  hClctCompMatch = fs->make<TH1F>("h_CLCT_match","h_CLCT_match",600,0.5,600.5);
  hClctCompMatch2 = fs->make<TH2F>("h_CLCT_match2","h_CLCT_match2",19,-9.5,9.5,36,0.5,36.5);
  hClctCompMatch2x = fs->make<TH2F>("h_CLCT_match2x","h_CLCT_match2x",19,-9.5,9.5,18,0.5,36.5);

  hClctCompTotal = fs->make<TH1F>("h_CLCT_total","h_CLCT_total",600,0.5,600.5);
  hClctCompTotal2 = fs->make<TH2F>("h_CLCT_total2","h_CLCT_total2",19,-9.5,9.5,36,0.5,36.5);
  hClctCompTotal2x = fs->make<TH2F>("h_CLCT_total2x","h_CLCT_total2x",19,-9.5,9.5,18,0.5,36.5);

  hLCTCompFound = fs->make<TH1F>("h_LCT_found","h_LCT_found",600,0.5,600.5);
  hLCTCompFound2 = fs->make<TH2F>("h_LCT_found2","h_LCT_found2",19,-9.5,9.5,36,0.5,36.5);
  hLCTCompFound2x = fs->make<TH2F>("h_LCT_found2x","h_LCT_found2x",19,-9.5,9.5,18,0.5,36.5);

  hLCTCompSameN = fs->make<TH1F>("h_LCT_SameN","h_LCT_SameN",600,0.5,600.5);
  hLCTCompSameN2 = fs->make<TH2F>("h_LCT_SameN2","h_LCT_SameN2",19,-9.5,9.5,36,0.5,36.5);
  hLCTCompSameN2x = fs->make<TH2F>("h_LCT_SameN2x","h_LCT_SameN2x",19,-9.5,9.5,18,0.5,36.5);

  hLCTCompMatch = fs->make<TH1F>("h_LCT_match","h_LCT_match",600,0.5,600.5);
  hLCTCompMatch2 = fs->make<TH2F>("h_LCT_match2","h_LCT_match2",19,-9.5,9.5,36,0.5,36.5);
  hLCTCompMatch2x = fs->make<TH2F>("h_LCT_match2x","h_LCT_match2x",19,-9.5,9.5,18,0.5,36.5);

  hLCTCompTotal = fs->make<TH1F>("h_LCT_total","h_LCT_total",600,0.5,600.5);
  hLCTCompTotal2 = fs->make<TH2F>("h_LCT_total2","h_LCT_total2",19,-9.5,9.5,36,0.5,36.5);
  hLCTCompTotal2x = fs->make<TH2F>("h_LCT_total2x","h_LCT_total2x",19,-9.5,9.5,18,0.5,36.5);

  //Chad's improved historgrams
  hAlctCompFound2i = fs->make<TH2F>("h_ALCT_found2i","h_ALCT_found2i",18,0,18,36,0.5,36.5);
  hAlctCompSameN2i = fs->make<TH2F>("h_ALCT_SameN2i","h_ALCT_SameN2i",18,0,18,36,0.5,36.5);
  hAlctCompMatch2i = fs->make<TH2F>("h_ALCT_match2i","h_ALCT_match2i",18,0,18,36,0.5,36.5);
  hAlctCompTotal2i = fs->make<TH2F>("h_ALCT_total2i","h_ALCT_total2i",18,0,18,36,0.5,36.5);
  hClctCompFound2i = fs->make<TH2F>("h_CLCT_found2i","h_CLCT_found2i",18,0,18,36,0.5,36.5);
  hClctCompSameN2i = fs->make<TH2F>("h_CLCT_SameN2i","h_CLCT_SameN2i",18,0,18,36,0.5,36.5);
  hClctCompMatch2i = fs->make<TH2F>("h_CLCT_match2i","h_CLCT_match2i",18,0,18,36,0.5,36.5);
  hClctCompTotal2i = fs->make<TH2F>("h_CLCT_total2i","h_CLCT_total2i",18,0,18,36,0.5,36.5);
  hLCTCompFound2i = fs->make<TH2F>("h_LCT_found2i","h_LCT_found2i",18,0,18,36,0.5,36.5);
  hLCTCompSameN2i = fs->make<TH2F>("h_LCT_SameN2i","h_LCT_SameN2i",18,0,18,36,0.5,36.5);
  hLCTCompMatch2i = fs->make<TH2F>("h_LCT_match2i","h_LCT_match2i",18,0,18,36,0.5,36.5);
  hLCTCompTotal2i = fs->make<TH2F>("h_LCT_total2i","h_LCT_total2i",18,0,18,36,0.5,36.5);

  //  hAlctCompFound = fs->make<TH1F>("h_ALCT_found","h_ALCT_found",600,0.5,600.5);

  // ALCTs.
  for (int i = 0; i < MAX_ENDCAPS; i++) { // endcaps
    for (int j = 0; j < CSC_TYPES; j++) { // station/ring
      char asdf[256];

      float csc_max = NCHAMBERS[j] + 0.5;
      if (i == 0) s = "Comp_ALCTs, " + csc_type_plus[j];
      else        s = "Comp_ALCTs, " + csc_type_minus[j];
      sprintf(asdf,"Comp_ALCTsFound_%i",i*CSC_TYPES+j);
      hAlctCompFoundCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"Comp_ALCTsSame_%i",i*CSC_TYPES+j);
      hAlctCompSameNCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"Comp_ALCTsTotal_%i",i*CSC_TYPES+j);
      hAlctCompTotalCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"Comp_ALCTsMatch_%i",i*CSC_TYPES+j);
      hAlctCompMatchCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      hAlctCompFoundCsc[i][j]->Sumw2();
      hAlctCompSameNCsc[i][j]->Sumw2();
      hAlctCompTotalCsc[i][j]->Sumw2();
      hAlctCompMatchCsc[i][j]->Sumw2();
    }
  }

  // CLCTs.
  for (int i = 0; i < MAX_ENDCAPS; i++) { // endcaps
    for (int j = 0; j < CSC_TYPES; j++) { // station/ring
      float csc_max = NCHAMBERS[j] + 0.5;
      char asdf[256];
      if (i == 0) s = "Comp_CLCTs, " + csc_type_plus[j];
      else        s = "Comp_CLCTs, " + csc_type_minus[j];
      sprintf(asdf,"Comp_CLCTsFound_%i",i*CSC_TYPES+j);
      hClctCompFoundCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"Comp_CLCTsSame_%i",i*CSC_TYPES+j);
      hClctCompSameNCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"Comp_CLCTsTotal_%i",i*CSC_TYPES+j);
      hClctCompTotalCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"Comp_CLCTsMatch_%i",i*CSC_TYPES+j);
      hClctCompMatchCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      hClctCompFoundCsc[i][j]->Sumw2();
      hClctCompSameNCsc[i][j]->Sumw2();
      hClctCompTotalCsc[i][j]->Sumw2();
      hClctCompMatchCsc[i][j]->Sumw2();
    }
  }

  // Correlated LCTs.
  for (int i = 0; i < MAX_ENDCAPS; i++) { // endcaps
    for (int j = 0; j < CSC_TYPES; j++) { // station/ring
      float csc_max = NCHAMBERS[j] + 0.5;
      char asdf[256];
      if (i == 0) s = "LCTs, " + csc_type_plus[j];
      else        s = "LCTs, " + csc_type_minus[j];
      sprintf(asdf,"LCTs_CompFound_%i",i*CSC_TYPES+j);
      hLctCompFoundCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"LCTs_CompSame_%i",i*CSC_TYPES+j);
      hLctCompSameNCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"LCTs_CompTotal_%i",i*CSC_TYPES+j);
      hLctCompTotalCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      sprintf(asdf,"LCTs_CompMatch_%i",i*CSC_TYPES+j);
      hLctCompMatchCsc[i][j] =
        fs->make<TH1F>(asdf, s.c_str(), NCHAMBERS[j], 0.5, csc_max);
      hLctCompFoundCsc[i][j]->Sumw2();
      hLctCompSameNCsc[i][j]->Sumw2();
      hLctCompTotalCsc[i][j]->Sumw2();
      hLctCompMatchCsc[i][j]->Sumw2();
    }
  }

  bookedCompHistos = true;
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               <<"bookedCompHistos  ";
  std::cout <<"bookCompHistos "<< std::endl;
}

void CSCTriggerPrimitivesReader::bookResolHistos() {
  edm::Service<TFileService> fs;

  // Limits for resolution histograms
  const double EDMIN = -0.05; // eta min
  const double EDMAX =  0.05; // eta max
  const double PDMIN = -5.0;  // phi min (mrad)
  const double PDMAX =  5.0;  // phi max (mrad)

  hResolDeltaWG = fs->make<TH1F>("", "Delta key wiregroup", 10, -5., 5.);

  hResolDeltaHS = fs->make<TH1F>("", "Delta key halfstrip", 10, -5., 5.);
  hResolDeltaDS = fs->make<TH1F>("", "Delta key distrip",   10, -5., 5.);

  hResolDeltaEta   = fs->make<TH1F>("", "#eta_rec-#eta_sim", 100, EDMIN, EDMAX);
  hResolDeltaPhi   = fs->make<TH1F>("", "#phi_rec-#phi_sim (mrad)", 100, -10., 10.);
  hResolDeltaPhiHS = fs->make<TH1F>("", "#phi_rec-#phi_sim (mrad), halfstrips", 100, -10., 10.);
  hResolDeltaPhiDS = fs->make<TH1F>("", "#phi_rec-#phi_sim (mrad), distrips", 100, -10., 10.);

  hEtaRecVsSim = fs->make<TH2F>("", "#eta_rec vs #eta_sim", 64, 0.9,  2.5,  64, 0.9,  2.5);
  hPhiRecVsSim = fs->make<TH2F>("", "#phi_rec vs #phi_sim", 100, 0., TWOPI, 100, 0., TWOPI);

  // LCT quantities per station
  char histname[60];
  for (int i = 0; i < MAX_STATIONS; i++) {
    sprintf(histname, "ALCTs vs eta, station %d", i+1);
    hAlctVsEta[i]    = fs->make<TH1F>("", histname, 66, 0.875, 2.525);

    sprintf(histname, "CLCTs vs phi, station %d", i+1);
    hClctVsPhi[i]    = fs->make<TH1F>("", histname, 100, 0.,   TWOPI);

    sprintf(histname, "#LT#eta_rec-#eta_sim#GT, station %d", i+1);
    hEtaDiffVsEta[i] = fs->make<TH1F>("", histname, 66, 0.875, 2.525);

    sprintf(histname, "#LT#phi_rec-#phi_sim#GT, station %d", i+1);
    hPhiDiffVsPhi[i] = fs->make<TH1F>("", histname, 100, 0.,   TWOPI);
  }

  for (int i = 0; i < CSC_TYPES; i++) {
    string t0 = "#eta_rec-#eta_sim, " + csc_type[i];
    hEtaDiffCsc[i][0] = fs->make<TH1F>("", t0.c_str(), 100, EDMIN, EDMAX);
    string t1 = t0 + ", endcap1";
    hEtaDiffCsc[i][1] = fs->make<TH1F>("", t1.c_str(), 100, EDMIN, EDMAX);
    string t2 = t0 + ", endcap2";
    hEtaDiffCsc[i][2] = fs->make<TH1F>("", t2.c_str(), 100, EDMIN, EDMAX);

    string t4 = "#eta_rec-#eta_sim vs wiregroup, " + csc_type[i];
    hEtaDiffVsWireCsc[i] =
      fs->make<TH2F>("", t4.c_str(), MAX_WG[i], 0., MAX_WG[i], 100, EDMIN, EDMAX);

    string u0 = "#phi_rec-#phi_sim, " + csc_type[i];
    hPhiDiffCsc[i][0] = fs->make<TH1F>("", u0.c_str(), 100, PDMIN, PDMAX);
    string u1 = u0 + ", endcap1";
    hPhiDiffCsc[i][1] = fs->make<TH1F>("", u1.c_str(), 100, PDMIN, PDMAX);
    string u2 = u0 + ", endcap2";
    hPhiDiffCsc[i][2] = fs->make<TH1F>("", u2.c_str(), 100, PDMIN, PDMAX);
    hPhiDiffCsc[i][3] = fs->make<TH1F>("", u0.c_str(), 100, PDMIN, PDMAX);
    hPhiDiffCsc[i][4] = fs->make<TH1F>("", u0.c_str(), 100, PDMIN, PDMAX);

    int MAX_DS = MAX_HS[i]/4;
    string u5 = "#phi_rec-#phi_sim (mrad) vs distrip, " + csc_type[i];
    hPhiDiffVsStripCsc[i][0] =
      fs->make<TH2F>("", u5.c_str(), MAX_DS,    0., MAX_DS,    100, PDMIN, PDMAX);
    string u6 = "#phi_rec-#phi_sim (mrad) vs halfstrip, " + csc_type[i];
    hPhiDiffVsStripCsc[i][1] =
      fs->make<TH2F>("", u6.c_str(), MAX_HS[i], 0., MAX_HS[i], 100, PDMIN, PDMAX);

    string u7 = "#phi(layer 1)-#phi(layer 6), mrad, " + csc_type[i];
    hTrueBendCsc[i] =fs->make<TH1F>("", u7.c_str(), 100, -10., 10.);
  }

  int max_patterns, phibend;
  max_patterns = CSCConstants::NUM_CLCT_PATTERNS;
  for (int i = 0; i < max_patterns; i++) {
    phibend = ptype_TMB07[i];
    sprintf(histname, "#phi_rec-#phi_sim, bend = %d", phibend);
    hPhiDiffPattern[i] = fs->make<TH1F>("", histname, 100, PDMIN, PDMAX);
  }

  bookedResolHistos = true;
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               <<"bookedResolHistos  ";
}

void CSCTriggerPrimitivesReader::bookEfficHistos() {
  edm::Service<TFileService> fs;

  // Efficiencies per station.
  char histname[60];
  for (int i = 0; i < MAX_STATIONS; i++) {
    sprintf(histname, "SimHits vs eta, station %d", i+1);
    hEfficHitsEta[i] = fs->make<TH1F>("", histname, 66, 0.875, 2.525);

    sprintf(histname, "ALCTs vs eta, station %d", i+1);
    hEfficALCTEta[i] = fs->make<TH1F>("", histname, 66, 0.875, 2.525);

    sprintf(histname, "CLCTs vs eta, station %d", i+1);
    hEfficCLCTEta[i] = fs->make<TH1F>("", histname, 66, 0.875, 2.525);
  }

  // Efficiencies per chamber type.
  for (int i = 0; i < CSC_TYPES; i++) {
    string t0 = "SimHits vs eta, " + csc_type[i];
    hEfficHitsEtaCsc[i] = fs->make<TH1F>("", t0.c_str(), 66, 0.875, 2.525);
    string t1 = "ALCTs vs eta, " + csc_type[i];
    hEfficALCTEtaCsc[i] = fs->make<TH1F>("", t1.c_str(), 66, 0.875, 2.525);
    string t2 = "CLCTs vs eta, " + csc_type[i];
    hEfficCLCTEtaCsc[i] = fs->make<TH1F>("", t1.c_str(), 66, 0.875, 2.525);
  }

  bookedEfficHistos = true;
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               <<"bookedEfficHistos  ";
  std::cout <<" bookedEfficHistos " << std::endl;
}

void CSCTriggerPrimitivesReader::fillALCTHistos(const CSCALCTDigiCollection* alcts) {
  // Book histos when called for the first time.
  if (!bookedALCTHistos) bookALCTHistos();

  int nValidALCTs = 0;
  for (auto detUnitIt = alcts->begin(); detUnitIt != alcts->end(); detUnitIt++) {
    int nValidALCTsPerCSC = 0;
    const CSCDetId& id = (*detUnitIt).first;
    if (checkBadChambers_ && badChambers_->isInBadChamber(id)) continue;
    const auto& range = (*detUnitIt).second;
    for (auto digiIt = range.first;
         digiIt != range.second; digiIt++) {

      bool alct_valid = (*digiIt).isValid();
      hAlctValid->Fill(alct_valid);
      if (alct_valid) {
        hAlctQuality->Fill((*digiIt).getQuality());
        hAlctAccel->Fill((*digiIt).getAccelerator());
        hAlctCollis->Fill((*digiIt).getCollisionB());
        hAlctKeyGroup->Fill((*digiIt).getKeyWG());
        hAlctBXN->Fill((*digiIt).getBX());

        int csctype = getCSCType(id);
        hAlctPerCSC->Fill(csctype);
        hAlctCsc[id.endcap()-1][csctype]->Fill(id.chamber());

        if (csctype==0) hAlctKeyGroupME11->Fill((*digiIt).getKeyWG());

        nValidALCTs++;
        nValidALCTsPerCSC++;

        if (debug) LogTrace("CSCTriggerPrimitivesReader")
                     << (*digiIt) << " found in ME" << ((id.endcap() == 1) ? "+" : "-")
                     << id.station() << "/" << id.ring() << "/" << id.chamber()
                     << " (sector " << id.triggerSector()
                     << " trig id. " << id.triggerCscId() << ")";
        //cout << "raw id = " << id.rawId() << endl;
      }
    }
    hAlctPerChamber->Fill(nValidALCTsPerCSC);
  }
  hAlctPerEvent->Fill(nValidALCTs);
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               << nValidALCTs << " valid ALCTs found in this event";
  numALCT += nValidALCTs;
}

void CSCTriggerPrimitivesReader::fillCLCTHistos(const CSCCLCTDigiCollection* clcts) {
  // Book histos when called for the first time.
  if (!bookedCLCTHistos) bookCLCTHistos();

  int nValidCLCTs = 0;
  for (auto detUnitIt = clcts->begin(); detUnitIt != clcts->end(); detUnitIt++) {
    int nValidCLCTsPerCSC = 0;
    const CSCDetId& id = (*detUnitIt).first;
    if (checkBadChambers_ && badChambers_->isInBadChamber(id)) continue;
    const auto& range = (*detUnitIt).second;
    for (auto digiIt = range.first;
         digiIt != range.second; digiIt++) {

      bool clct_valid = (*digiIt).isValid();
      hClctValid->Fill(clct_valid);
      if (clct_valid) {
        int striptype = (*digiIt).getStripType();
        int keystrip  = (*digiIt).getKeyStrip(); // halfstrip #
        if (striptype == 0) keystrip /= 4;       // distrip # for distrip ptns
        hClctQuality->Fill((*digiIt).getQuality());
        hClctStripType->Fill(striptype);
        hClctSign->Fill((*digiIt).getBend());
        hClctCFEB->Fill((*digiIt).getCFEB());
        hClctBXN->Fill((*digiIt).getBX());
        hClctKeyStrip[striptype]->Fill(keystrip);
        hClctPattern[striptype]->Fill((*digiIt).getPattern());

        int csctype = getCSCType(id);
        hClctPerCSC->Fill(csctype);
        hClctCsc[id.endcap()-1][csctype]->Fill(id.chamber());

        if (striptype==1 && csctype==0) hClctKeyStripME11->Fill(keystrip);

        int phibend;
        int pattern = (*digiIt).getPattern();
        phibend = ptype_TMB07[pattern];
        hClctBendCsc[csctype][striptype]->Fill(abs(phibend));

        if (striptype == 0) // distrips
          hClctKeyStripCsc[csctype]->Fill(keystrip);

        nValidCLCTs++;
        nValidCLCTsPerCSC++;

        if (debug) LogTrace("CSCTriggerPrimitivesReader")
                     << (*digiIt) << " found in ME" << ((id.endcap() == 1) ? "+" : "-")
                     << id.station() << "/" << id.ring() << "/" << id.chamber()
                     << " (sector " << id.triggerSector()
                     << " trig id. " << id.triggerCscId() << ")";
      }
    }
    hClctPerChamber->Fill(nValidCLCTsPerCSC);
  }
  hClctPerEvent->Fill(nValidCLCTs);
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               << nValidCLCTs << " valid CLCTs found in this event";
  numCLCT += nValidCLCTs;
}

void CSCTriggerPrimitivesReader::fillLCTTMBHistos(const CSCCorrelatedLCTDigiCollection* lcts) {
  // Book histos when called for the first time.
  if (!bookedLCTTMBHistos) bookLCTTMBHistos();

  int nValidLCTs = 0;
  bool alct_valid, clct_valid;

  for (auto detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) {
    int nValidLCTsPerCSC = 0;
    const CSCDetId& id = (*detUnitIt).first;
    if (checkBadChambers_ && badChambers_->isInBadChamber(id)) continue;
    const auto& range = (*detUnitIt).second;
    for (auto digiIt = range.first;
         digiIt != range.second; digiIt++) {

      bool lct_valid = (*digiIt).isValid();
      hLctTMBValid->Fill(lct_valid);
      if (lct_valid) {
        hLctTMBEndcap->Fill(id.endcap());
        hLctTMBStation->Fill(id.station());
        hLctTMBSector->Fill(id.triggerSector());
        hLctTMBRing->Fill(id.ring());
        hLctTMBChamber[id.station()-1]->Fill(id.triggerCscId());

        int quality = (*digiIt).getQuality();
        hLctTMBQuality->Fill(quality);
        hLctTMBBXN->Fill((*digiIt).getBX());

        alct_valid = (quality != 0 && quality != 2);
        if (alct_valid) {
          hLctTMBKeyGroup->Fill((*digiIt).getKeyWG());
        }

        clct_valid = (quality != 0 && quality != 1);
        if (clct_valid) {
          hLctTMBKeyStrip->Fill((*digiIt).getStrip());
          hLctTMBStripType->Fill(1.);
          hLctTMBPattern->Fill((*digiIt).getPattern());
          hLctTMBBend->Fill((*digiIt).getBend());
        }

        int csctype = getCSCType(id);
        hLctTMBPerCSC->Fill(csctype);
        hLctTMBCsc[id.endcap()-1][csctype]->Fill(id.chamber());
        // Truly correlated LCTs; for DAQ
        if (alct_valid && clct_valid) hCorrLctTMBPerCSC->Fill(csctype);

        if (alct_valid && csctype==0) {
          hLctTMBKeyGroupME11->Fill((*digiIt).getKeyWG());
        }
        if (clct_valid && csctype==0) {
          hLctTMBKeyStripME11->Fill((*digiIt).getStrip());
        }

        nValidLCTs++;
        nValidLCTsPerCSC++;

        if (debug) LogTrace("CSCTriggerPrimitivesReader")
                     << (*digiIt) << " found in ME" << ((id.endcap() == 1) ? "+" : "-")
                     << id.station() << "/" << id.ring() << "/" << id.chamber()
                     << " (sector " << id.triggerSector()
                     << " trig id. " << id.triggerCscId() << ")";
      }
    }
    hLctTMBPerChamber->Fill(nValidLCTsPerCSC);
  }
  hLctTMBPerEvent->Fill(nValidLCTs);
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               << nValidLCTs << " valid LCTs found in this event";
  numLCTTMB += nValidLCTs;
}

void CSCTriggerPrimitivesReader::fillLCTMPCHistos(const CSCCorrelatedLCTDigiCollection* lcts) {
  // Book histos when called for the first time.
  if (!bookedLCTMPCHistos) bookLCTMPCHistos();

  int nValidLCTs = 0;
  bool alct_valid, clct_valid;

  for (auto detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) {
    const CSCDetId& id = (*detUnitIt).first;
    if (checkBadChambers_ && badChambers_->isInBadChamber(id)) continue;
    const auto& range = (*detUnitIt).second;
    for (auto digiIt = range.first;
         digiIt != range.second; digiIt++) {

      bool lct_valid = (*digiIt).isValid();
      hLctMPCValid->Fill(lct_valid);
      if (lct_valid) {
        hLctMPCEndcap->Fill(id.endcap());
        hLctMPCStation->Fill(id.station());
        hLctMPCSector->Fill(id.triggerSector());
        hLctMPCRing->Fill(id.ring());
        hLctMPCChamber[id.station()-1]->Fill(id.triggerCscId());

        int quality = (*digiIt).getQuality();
        hLctMPCQuality->Fill(quality);
        hLctMPCBXN->Fill((*digiIt).getBX());

        alct_valid = (quality != 0 && quality != 2);
        if (alct_valid) {
          hLctMPCKeyGroup->Fill((*digiIt).getKeyWG());
        }

        clct_valid = (quality != 0 && quality != 1);
        if (clct_valid) {
          hLctMPCKeyStrip->Fill((*digiIt).getStrip());
          hLctMPCStripType->Fill(1.);
          hLctMPCPattern->Fill((*digiIt).getPattern());
          hLctMPCBend->Fill((*digiIt).getBend());
        }

        int csctype = getCSCType(id);
        hLctMPCPerCSC->Fill(csctype);
        // Truly correlated LCTs; for DAQ
        if (alct_valid && clct_valid) hCorrLctMPCPerCSC->Fill(csctype);


        if (alct_valid && csctype==0) {
          hLctMPCKeyGroupME11->Fill((*digiIt).getKeyWG());
        }
        if (clct_valid && csctype==0) {
          hLctMPCKeyStripME11->Fill((*digiIt).getStrip());
        }

        nValidLCTs++;

        if (debug) LogTrace("CSCTriggerPrimitivesReader")
                     << "MPC "
                     << (*digiIt) << " found in ME" << ((id.endcap() == 1) ? "+" : "-")
                     << id.station() << "/" << id.ring() << "/" << id.chamber()
                     << " (sector " << id.triggerSector()
                     << " trig id. " << id.triggerCscId() << ")";
      }
    }
  }
  hLctMPCPerEvent->Fill(nValidLCTs);
  if (debug) LogTrace("CSCTriggerPrimitivesReader")
               << nValidLCTs << " MPC LCTs found in this event";
  numLCTMPC += nValidLCTs;
}

void CSCTriggerPrimitivesReader::compare(const CSCALCTDigiCollection* alcts_data,
                                         const CSCALCTDigiCollection* alcts_emul,
                                         const CSCCLCTDigiCollection* clcts_data,
                                         const CSCCLCTDigiCollection* clcts_emul,
                                         const CSCCLCTPreTriggerDigiCollection* pretrigs_emul,
                                         const CSCCorrelatedLCTDigiCollection* lcts_data,
                                         const CSCCorrelatedLCTDigiCollection* lcts_emul){

  // Book histos when called for the first time.
  if (!bookedCompHistos) bookCompHistos();

  // Comparisons
  compareALCTs(alcts_data, alcts_emul);
  compareCLCTs(clcts_data, clcts_emul, pretrigs_emul);
  compareLCTs(lcts_data,  lcts_emul, alcts_data, clcts_data);
  //compareMPCLCTs(mpclcts_data,  mpclcts_emul, alcts_data, clcts_data);
}

void CSCTriggerPrimitivesReader::compareALCTs(const CSCALCTDigiCollection* alcts_data,
                                              const CSCALCTDigiCollection* alcts_emul) {
  int emul_corr_bx;

  // Should be taken from config. parameters.
  int fifo_pretrig     = 10;
  int fpga_latency     =  6;
  int l1a_window_width =  7;
  // Time offset of raw hits w.r.t. the full 12-bit BXN.
  int rawhit_tbin_offset =
    (fifo_pretrig - fpga_latency) + (l1a_window_width-1)/2;
  // Extra difference due to additional register stages; determined
  // empirically.
  int register_delay =  2;
  // Loop over all chambers in search for ALCTs.
  std::vector<CSCALCTDigi>::const_iterator pd, pe;
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= maxRing(stat); ring++) {
        for (int cham = 1; cham <= 36; cham++) {
          // Calculate DetId.  0th layer means whole chamber.
          CSCDetId detid(endc, stat, ring, cham, 0);

          // Skip chambers marked as bad.
          if (checkBadChambers_ && badChambers_->isInBadChamber(detid)) continue;
          std::vector<CSCALCTDigi> alctV_data, alctV_emul;
          const auto& drange = alcts_data->get(detid);
          for (auto digiIt = drange.first; digiIt != drange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              alctV_data.push_back(*digiIt);
            }
          }

          const auto& erange = alcts_emul->get(detid);
          std::vector<bool> bookedalctV_emul;
          for (auto digiIt = erange.first; digiIt != erange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              alctV_emul.push_back(*digiIt);
              bookedalctV_emul.push_back(false);
            }
          }

          int ndata = alctV_data.size();
          int nemul = alctV_emul.size();

          if (ndata == 0 && nemul == 0) continue;
          bool firstfill = true;

          if (debug or (stat == 1 and ring== 1 and (ndata != nemul)) ) {
            ostringstream strstrm;
            strstrm << "\n--- ME" << ((detid.endcap() == 1) ? "+" : "-")
                    << detid.station() << "/" << detid.ring() << "/"
                    << detid.chamber()
                    << " (sector "  << detid.triggerSector()
                    << " trig id. " << detid.triggerCscId() << "):\n";
            strstrm << "  **** " << ndata << " valid data ALCTs found:\n";
            for (pd = alctV_data.begin(); pd != alctV_data.end(); pd++) {
              strstrm << "     " << (*pd)
                      << " Full BX = " << (*pd).getFullBX() << "\n";
            }
            strstrm << "  **** " << nemul << " valid emul ALCTs found:\n";
            for (pe = alctV_emul.begin(); pe != alctV_emul.end(); pe++) {
              strstrm << "     " << (*pe);
              for (pd = alctV_data.begin(); pd != alctV_data.end(); pd++) {
                if ((*pd).getTrknmb() == (*pe).getTrknmb()) {
                  int emul_bx = (*pe).getBX();
                  emul_corr_bx =
                    emul_bx - rawhit_tbin_offset + register_delay;
                  strstrm << " Corr BX = " << emul_corr_bx;
                  break;
                }
              }
              strstrm << "\n";
            }
            LogTrace("CSCTriggerPrimitivesReader") << strstrm.str();
            //if (stat==1 && ring==1)
            //	std::cout <<"ME11  CompareALCTs "<< strstrm.str()<< std::endl;
          }

          //Data, add HS quality later
          int perEv_nStub_data = 0;
          int perEv_nStub_emul = 0;
          perStub[0].init(RUN_, Event_);
          perStub[1].init(RUN_, Event_);
          for (pd = alctV_data.begin(); pd != alctV_data.end(); pd++){
            perEv_nStub_data++;
          }
          perStub[0].t_endcap = endc;
          perStub[0].t_station = stat;
          perStub[0].t_chamber = cham;
          perStub[0].t_ring = ring;
          perStub[0].t_EventNumberAnalyzed = eventsAnalyzed;
          perStub[0].t_nStubs              = ndata;
          perStub[0].t_nStubs_readout              = ndata;
          event_tree[0]->Fill();
          //Emul
          for (pe = alctV_emul.begin(); pe != alctV_emul.end(); pe++){
            perEv_nStub_emul++;
          }
          perStub[1].t_endcap = endc;
          perStub[1].t_station = stat;
          perStub[1].t_chamber = cham;
          perStub[1].t_ring = ring;
          perStub[1].t_EventNumberAnalyzed = eventsAnalyzed;
          perStub[1].t_nStubs              = nemul;
          perStub[1].t_nStubs_readout      = nemul;
          event_tree[1]->Fill();

          int csctype = getCSCType(detid);
          hAlctCompFoundCsc[endc-1][csctype]->Fill(cham);
          int mychamber = chamberSerial(detid);
          hAlctCompFound->Fill(mychamber);
          int ix = chamberIX(detid);
          int ix2 = chamberIXi(detid);
          //	  printf("station %i, ring %i, chamber %i, ix+(detid.ring()-1) %i\n",
          //		 detid.station(),detid.ring(),detid.chamber(),ix);
          if(detid.station()>1 && detid.ring()==1) {
            hAlctCompFound2x->Fill(ix,detid.chamber()*2);
          }
          else {
            hAlctCompFound2->Fill(ix,detid.chamber());
          }
          hAlctCompFound2i->Fill(ix2,detid.chamber());

          if (ndata != nemul) {
            LogTrace("CSCTriggerPrimitivesReader")
              << "   +++ Different numbers of ALCTs found in ME"
              << ((endc == 1) ? "+" : "-") << stat << "/"
              << ring << "/" << cham
              << ": data = " << ndata << " emulator = " << nemul << " +++\n";
          }
          else {
            hAlctCompSameNCsc[endc-1][csctype]->Fill(cham);
            if(detid.station()>1 && detid.ring()==1) {
              hAlctCompSameN2x->Fill(ix,detid.chamber()*2);
            }
            else {
              hAlctCompSameN2->Fill(ix,detid.chamber());
            }
            hAlctCompSameN2i->Fill(ix2,detid.chamber());
          }

          for (int i = 0; i < ndata; i++) {
            if (alctV_data[i].isValid() == 0) continue;
            int data_trknmb    = alctV_data[i].getTrknmb();
            //int data_quality   = alctV_data[i].getQuality();
            //int data_accel     = alctV_data[i].getAccelerator();
            //int data_collB     = alctV_data[i].getCollisionB();
            int data_wiregroup = alctV_data[i].getKeyWG();
            int data_bx        = alctV_data[i].getBX();

            stubs_comparison[0].init(RUN_, Event_);
            stubs_comparison[0].firstfill = firstfill;
            if (firstfill) firstfill = false;
            stubs_comparison[0].nEvents = eventsAnalyzed;
            stubs_comparison[0].endcap = endc;
            stubs_comparison[0].station = stat;
            stubs_comparison[0].ring = ring;
            stubs_comparison[0].chamber = cham;
            stubs_comparison[0].chambertype = detid.iChamberType();
            stubs_comparison[0].totStubs_data = ndata;
            stubs_comparison[0].totStubs_emul = nemul;
            stubs_comparison[0].totStubs_emul_readout = nemul;
            stubs_comparison[0].nStub_data = i+1;
            stubs_comparison[0].has_data = true;
            stubs_comparison[0].quality_data = alctV_data[i].getQuality();
            stubs_comparison[0].key_WG_data = alctV_data[i].getKeyWG();
            stubs_comparison[0].bx_data = data_bx;
            stubs_comparison[0].fullbx_data = alctV_data[i].getFullBX();
            stubs_comparison[0].trknmb_data = data_trknmb;
            GlobalPoint gp_alct_data(getGlobalPosition(detid.rawId(), data_wiregroup, 60));
            stubs_comparison[0].eta_data = gp_alct_data.eta();
            for (int j=0; j<nemul; j++){
              if (alctV_emul[j].isValid() == 0) continue;
              if (bookedalctV_emul[j]) continue; //used alct

              int emul_trknmb    = alctV_emul[j].getTrknmb();
              //int emul_quality   = alctV_emul[j].getQuality();
              //int emul_accel     = alctV_emul[j].getAccelerator();
              //int emul_collB     = alctV_emul[j].getCollisionB();
              int emul_wiregroup = alctV_emul[j].getKeyWG();
              int emul_bx        = alctV_emul[j].getBX();
              // Emulator BX re-calculated for comparison with BX in the data.
              emul_corr_bx = emul_bx - rawhit_tbin_offset + register_delay;
              if (dataIsAnotherMC_)
                emul_corr_bx = emul_bx;

              if (ndata == nemul) {
                hAlctCompTotal->Fill(mychamber);
                hAlctCompTotalCsc[endc-1][csctype]->Fill(cham);
                if(detid.station()>1 && detid.ring()==1) {
                  hAlctCompTotal2x->Fill(ix,detid.chamber()*2);
                }
                else {
                  hAlctCompTotal2->Fill(ix,detid.chamber());
                }
                hAlctCompTotal2i->Fill(ix2,detid.chamber());
              }
              if (data_trknmb    == emul_trknmb )  {
                if (abs(data_wiregroup - emul_wiregroup) <= 2){
                  hAlctCompMatchCsc[endc-1][csctype]->Fill(cham);
                  hAlctCompMatch->Fill(mychamber);
                  if(detid.station()>1 && detid.ring()==1) {
                    hAlctCompMatch2x->Fill(ix,detid.chamber()*2);
                  }
                  else {
                    hAlctCompMatch2->Fill(ix,detid.chamber());
                  }
                  hAlctCompMatch2i->Fill(ix2,detid.chamber());
                }
                if (debug) LogTrace("CSCTriggerPrimitivesReader")
                             << "       Identical ALCTs #" << data_trknmb;
                stubs_comparison[0].nStub_emul = j+1;
                stubs_comparison[0].has_emul = true;
                stubs_comparison[0].quality_emul = alctV_emul[j].getQuality();
                stubs_comparison[0].key_WG_emul = alctV_emul[j].getKeyWG();
                stubs_comparison[0].bx_emul = alctV_emul[j].getBX();
                stubs_comparison[0].trknmb_emul = emul_trknmb;
                GlobalPoint gp_alct_emul(getGlobalPosition(detid.rawId(), emul_wiregroup, 60));
                stubs_comparison[0].eta_emul = gp_alct_emul.eta();
                stubs_comparison[0].bx_corr_emul = emul_corr_bx;
                bookedalctV_emul[j] = true;
                break;
              }
              else {
                LogTrace("CSCTriggerPrimitivesReader")
                  << "       Different ALCTs #" << data_trknmb << " in ME"
                  << ((endc == 1) ? "+" : "-") << stat << "/"
                  << ring << "/" << cham;
              }

            }//loop emul
            if (debug and stubs_comparison[0].key_WG_data != stubs_comparison[0].key_WG_emul)
              LogTrace("CSCTriggerPrimitivesReader")
                <<" not matched ALCT from data "<< alctV_data[i]
                <<" stubs_comparison 0 key_WG_data "<<stubs_comparison[0].key_WG_data <<" key_WG_emul "<< stubs_comparison[0].key_WG_emul;
            //if (stat==1) std::cout <<" stub_tree filled , ring "<< stubs_comparison[0].ring << std::endl;
            //cout <<"ALCT data BX "<< stubs_comparison[0].bx_data <<" WG "<< stubs_comparison[0].key_WG_data <<" emul BX "<< stubs_comparison[0].bx_emul<<" emul BX corrected "<< stubs_comparison[0].bx_corr_emul <<" WG "<< stubs_comparison[0].key_WG_emul << endl;
            stub_tree[0]->Fill();
          }//loop data
          for (int i = 0; i<nemul; i++){
            if (alctV_emul[i].isValid() == 0 or bookedalctV_emul[i]) continue;
            int emul_bx        = alctV_emul[i].getBX();
            stubs_comparison[0].init(RUN_, Event_);
            stubs_comparison[0].firstfill = firstfill;
            if (firstfill) firstfill = false;
            stubs_comparison[0].nEvents = eventsAnalyzed;
            stubs_comparison[0].endcap = endc;
            stubs_comparison[0].station = stat;
            stubs_comparison[0].ring = ring;
            stubs_comparison[0].chamber = cham;
            stubs_comparison[0].chambertype = detid.iChamberType();
            stubs_comparison[0].totStubs_data = ndata;
            stubs_comparison[0].totStubs_emul = nemul;
            stubs_comparison[0].totStubs_emul_readout = nemul;
            stubs_comparison[0].nStub_data = -1;
            stubs_comparison[0].nStub_emul = i+1;
            stubs_comparison[0].has_data = false;
            stubs_comparison[0].has_emul = true;
            stubs_comparison[0].trknmb_emul = alctV_emul[i].getTrknmb();
            stubs_comparison[0].quality_emul = alctV_emul[i].getQuality();
            stubs_comparison[0].key_WG_emul = alctV_emul[i].getKeyWG();
            stubs_comparison[0].bx_emul = alctV_emul[i].getBX();
            stubs_comparison[0].fullbx_emul = alctV_emul[i].getFullBX();
            GlobalPoint gp_alct_emul(getGlobalPosition(detid.rawId(), alctV_emul[i].getKeyWG(), 60));
            stubs_comparison[0].eta_emul = gp_alct_emul.eta();
            // Emulator BX re-calculated for comparison with BX in the data.
            emul_corr_bx = emul_bx - rawhit_tbin_offset + register_delay;
            if (dataIsAnotherMC_)
              emul_corr_bx = emul_bx;
            stubs_comparison[0].bx_corr_emul = emul_corr_bx;
            stub_tree[0]->Fill();
            if (debug)
              LogTrace("CSCTriggerPrimitivesReader")
                <<"not matched ALCT from emulation "<< alctV_emul[i];

          }//loop emul
        }
      }
    }
  }
}// compare ALCT

void CSCTriggerPrimitivesReader::compareCLCTs(const CSCCLCTDigiCollection* clcts_data,
                                              const CSCCLCTDigiCollection* clcts_emul,
                                              const CSCCLCTPreTriggerDigiCollection* pretrigs_emul) {
  // Number of Tbins before pre-trigger for raw cathode hits.
  const int tbin_cathode_offset = 7;
  //const int tbin_cathode_offset = 8;//in MC, it became 8, Tao
  const int pretrig_trig_zone = 5;// max distance between CLCT key hs and pretrigger hs

  // Loop over all chambers in search for CLCTs.
  std::vector<CSCCLCTDigi>::const_iterator pd, pe;
  std::vector<CSCCLCTPreTriggerDigi>::const_iterator pretrig;
  perStub[2].init(RUN_, Event_);
  perStub[3].init(RUN_, Event_);
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= maxRing(stat); ring++) {
        for (int cham = 1; cham <= 36; cham++) {
          // Calculate DetId.  0th layer means whole chamber.
          CSCDetId detid(endc, stat, ring, cham, 0);

          // Skip chambers marked as bad.
          if (checkBadChambers_ && badChambers_->isInBadChamber(detid)) continue;

          std::vector<CSCCLCTDigi> clctV_data, clctV_emul;
          const auto& drange = clcts_data->get(detid);
          for (auto digiIt = drange.first; digiIt != drange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              clctV_data.push_back(*digiIt);
            }
          }

          const auto& erange = clcts_emul->get(detid);
          std::vector<bool> bookedclctV_emul;
          for (auto digiIt = erange.first; digiIt != erange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              for (auto pclct : clctV_emul){
                if (digiIt->getBX() != pclct.getBX() and abs(digiIt->getBX() - pclct.getBX())< 5)
                  LogTrace("CSCTriggerPrimitivesReader") <<"Two CLCTs very close in timing!!! Special event: first clct "<< pclct <<" second clct "<<*digiIt << std::endl;
              }
              clctV_emul.push_back(*digiIt);
              bookedclctV_emul.push_back(false);
            }
          }

          std::vector<CSCCLCTPreTriggerDigi> pretrigV_emul;
          const auto& pretrigrange = pretrigs_emul->get(detid);
          for (auto pretrigIt = pretrigrange.first; pretrigIt != pretrigrange.second; pretrigIt++){
            if ((*pretrigIt).isValid()){
              pretrigV_emul.push_back(*pretrigIt);
            }
          }

          int ndata = clctV_data.size();
          int nemul = clctV_emul.size();
          if (ndata == 0 && nemul == 0) continue;
          bool firstfill = true;
          int nemul_readout = 0;
          for (pe = clctV_emul.begin(); pe != clctV_emul.end(); pe++) {
            for (pd = clctV_data.begin(); pd != clctV_data.end(); pd++) {
              int emul_bx = (*pe).getBX();
              int corr_bx =
                ((*pd).getFullBX() + emul_bx - tbin_cathode_offset) & 0x03;
              int bx_data = pd->getBX();
              //if (corr_bx == bx_data or abs(pe->getKeyStrip() - pd->getKeyStrip())<=1){//if emulated BX after correction is same as data bx, it will be readout
              if (corr_bx == bx_data){//if emulated BX after correction is same as data bx, it will be readout
                nemul_readout++;
                break;
              }
            }
          }

          if (debug or nemul>ndata or (ndata != nemul) or ndata != nemul_readout) {
            LogTrace("CSCTriggerPrimitivesReader") << " CLCTs from data "<< ndata <<" CLCTs from emul "<< nemul <<" readout "<< nemul_readout;
            ostringstream strstrm;
            strstrm << "\n--- ME" << ((detid.endcap() == 1) ? "+" : "-")
                    << detid.station() << "/" << detid.ring() << "/"
                    << detid.chamber()
                    << " (sector "  << detid.triggerSector()
                    << " trig id. " << detid.triggerCscId() << "):\n";
            strstrm << "  **** " << ndata << " valid data CLCTs found:\n";
            for (pd = clctV_data.begin(); pd != clctV_data.end(); pd++) {
              strstrm << "     " << (*pd)
                      << " Full BX = " << (*pd).getFullBX() << "\n";
            }
            strstrm << "  **** " << nemul << " valid emul CLCTs found:\n";
            for (pe = clctV_emul.begin(); pe != clctV_emul.end(); pe++) {
              strstrm << "     " << (*pe);
              for (pd = clctV_data.begin(); pd != clctV_data.end(); pd++) {
                if ((*pd).getTrknmb() == (*pe).getTrknmb() or abs((*pe).getKeyStrip() - (*pd).getKeyStrip())<5) {
                  int emul_bx = (*pe).getBX();
                  int corr_bx =
                    ((*pd).getFullBX() + emul_bx - tbin_cathode_offset) & 0x03;
                  strstrm << " Corr BX = " << corr_bx;
                  break;
                }
              }
              strstrm << "\n";
            }
            LogTrace("CSCTriggerPrimitivesReader") << strstrm.str();
          }
          if (nemul_readout > 2){ //reduce nemul_readout to 2 by hand
            //cout <<"CLCT matching nemul readout is "<< nemul_readout <<", larger than 2. reduce it to 2 by hand"<<endl;
            nemul_readout = 2;
          }


          //Per event Fill, From Luca
          //Data, add HS quality later
          int perEv_nStub_data = 0;
          int perEv_nStub_emul = 0;
          perStub[2].init(RUN_, Event_);
          perStub[3].init(RUN_, Event_);
          for (pd = clctV_data.begin(); pd != clctV_data.end(); pd++){
            perEv_nStub_data++;
          }
          perStub[2].t_endcap = endc;
          perStub[2].t_station = stat;
          perStub[2].t_chamber = cham;
          perStub[2].t_ring = ring;
          perStub[2].t_EventNumberAnalyzed = eventsAnalyzed;
          perStub[2].t_nStubs              = ndata;
          perStub[2].t_nStubs_readout              = ndata;
          event_tree[2]->Fill();
          //Emul
          for (pe = clctV_emul.begin(); pe != clctV_emul.end(); pe++){
            perEv_nStub_emul++;
          }
          perStub[3].t_endcap = endc;
          perStub[3].t_station = stat;
          perStub[3].t_chamber = cham;
          perStub[3].t_ring = ring;
          perStub[3].t_EventNumberAnalyzed = eventsAnalyzed;
          perStub[3].t_nStubs              = nemul;
          perStub[3].t_nStubs_readout      = nemul_readout;
          event_tree[3]->Fill();


          int csctype = getCSCType(detid);
          hClctCompFoundCsc[endc-1][csctype]->Fill(cham);
          int ix = chamberIX(detid);
          int ix2 = chamberIXi(detid);
          if(detid.station()>1 && detid.ring()==1) {
            hClctCompFound2x->Fill(ix,detid.chamber()*2);
          }
          else {
            hClctCompFound2->Fill(ix,detid.chamber());
          }
          hClctCompFound2i->Fill(ix2,detid.chamber());
          if (ndata != nemul) {
            LogTrace("CSCTriggerPrimitivesReader")
              //cerr
              << "   +++ Different numbers of CLCTs found in ME"
              << ((endc == 1) ? "+" : "-") << stat << "/"
              << ring << "/" << cham
              << ": data = " << ndata << " emulator = " << nemul << " +++\n";
          }
          else {
            hClctCompSameNCsc[endc-1][csctype]->Fill(cham);
            if(detid.station()>1 && detid.ring()==1) {
              hClctCompSameN2x->Fill(ix,detid.chamber()*2);
            }
            else {
              hClctCompSameN2->Fill(ix,detid.chamber());
            }
            hClctCompSameN2i->Fill(ix2,detid.chamber());
          }
          int i=-1;
          int testwg = 20;
          for (pd = clctV_data.begin(); pd != clctV_data.end(); pd++) {
            i++;
            if ((*pd).isValid() == 0) continue;
            int data_trknmb    = (*pd).getTrknmb();
            int data_quality   = (*pd).getQuality();
            int data_pattern   = (*pd).getPattern();
            int data_striptype = (*pd).getStripType();
            int data_bend      = (*pd).getBend();
            int data_keystrip  = (*pd).getKeyStrip();
            int data_cfeb      = (*pd).getCFEB();
            int data_bx        = (*pd).getBX();
            int fullBX = (*pd).getFullBX(); // 12-bit full BX

            if (data_keystrip >= 128 and stat == 1 and ring == 1) testwg=5;
            else testwg = 20;

            stubs_comparison[1].init(RUN_, Event_);
            stubs_comparison[1].firstfill = firstfill;
            if (firstfill) firstfill = false;
            stubs_comparison[1].nEvents = eventsAnalyzed;
            stubs_comparison[1].endcap = endc;
            stubs_comparison[1].station = stat;
            stubs_comparison[1].ring = ring;
            stubs_comparison[1].chamber = cham;
            stubs_comparison[1].chambertype = detid.iChamberType();
            stubs_comparison[1].totStubs_data = ndata;
            stubs_comparison[1].totStubs_emul = nemul;
            stubs_comparison[1].totStubs_emul_readout = nemul_readout;
            stubs_comparison[1].nStub_data = i+1;
            stubs_comparison[1].has_data = true;
            stubs_comparison[1].quality_data = (*pd).getQuality();
            stubs_comparison[1].key_hs_data = (*pd).getKeyStrip();
            stubs_comparison[1].bend_data = (*pd).getBend();
            stubs_comparison[1].pattern_data = (*pd).getPattern();
            stubs_comparison[1].bx_data = (*pd).getBX();
            stubs_comparison[1].fullbx_data = (*pd).getFullBX();
            stubs_comparison[1].trknmb_data = data_trknmb;
            GlobalPoint gp_clct_data(getGlobalPosition(detid.rawId(), testwg, (*pd).getKeyStrip()));
            stubs_comparison[1].phi_data = gp_clct_data.phi();

            int j=-1;
            for (pe = clctV_emul.begin(); pe != clctV_emul.end(); pe++) {
              j++;
              if ((*pe).isValid() == 0) continue;
              if (bookedclctV_emul[j]) continue; //used alct
              int emul_trknmb    = (*pe).getTrknmb();
              int emul_quality   = (*pe).getQuality();
              int emul_pattern   = (*pe).getPattern();
              int emul_striptype = (*pe).getStripType();
              int emul_bend      = (*pe).getBend();
              int emul_keystrip  = (*pe).getKeyStrip();
              int emul_cfeb      = (*pe).getCFEB();
              int emul_bx        = (*pe).getBX();

              if (abs(data_keystrip - emul_keystrip) <= 2){
                // Emulator BX re-calculated using 12-bit full BX number.
                // Used for comparison with BX in the data.
                int emul_corr_bx =
                  (fullBX + emul_bx - tbin_cathode_offset) & 0x03;
                //std::cout <<"CLCT data_bx "<< data_bx <<" emul_corr_bx "<< emul_corr_bx << std::endl;
                if (dataIsAnotherMC_)
                  emul_corr_bx = (emul_bx & 0x03);
                if (ndata == nemul) {
                  hClctCompTotalCsc[endc-1][csctype]->Fill(cham);
                  if(detid.station()>1 && detid.ring()==1) {
                    hClctCompTotal2x->Fill(ix,detid.chamber()*2);
                  }
                  else {
                    hClctCompTotal2->Fill(ix,detid.chamber());
                  }
                  hClctCompTotal2i->Fill(ix2,detid.chamber());
                }
                if (data_quality   == emul_quality   &&
                    data_pattern   == emul_pattern   &&
                    data_striptype == emul_striptype &&
                    data_bend      == emul_bend      &&
                    data_keystrip  == emul_keystrip  &&
                    data_cfeb      == emul_cfeb      ){
                  if (ndata == nemul) {
                    hClctCompMatchCsc[endc-1][csctype]->Fill(cham);
                    if(detid.station()>1 && detid.ring()==1) {
                      hClctCompMatch2x->Fill(ix,detid.chamber()*2);
                    }
                    else {
                      hClctCompMatch2->Fill(ix,detid.chamber());
                    }
                    hClctCompMatch2i->Fill(ix2,detid.chamber());
                  }
                  if (debug) LogTrace("CSCTriggerPrimitivesReader")
                               << "       Identical CLCTs #" << data_trknmb;
                }
                else {
                  LogTrace("CSCTriggerPrimitivesReader")
                    << "       Different CLCTs #" << data_trknmb << " in ME"
                    << ((endc == 1) ? "+" : "-") << stat << "/"
                    << ring << "/" << cham <<" data_bx "<< data_bx <<" emul_corr_bx "<< emul_corr_bx;
                }
                stubs_comparison[1].nStub_emul = j+1;
                stubs_comparison[1].has_emul = true;
                stubs_comparison[1].quality_emul = (*pe).getQuality();
                stubs_comparison[1].key_hs_emul = (*pe).getKeyStrip();
                stubs_comparison[1].bend_emul = (*pe).getBend();
                stubs_comparison[1].pattern_emul = (*pe).getPattern();
                stubs_comparison[1].bx_emul = (*pe).getBX();
                stubs_comparison[1].bx_corr_emul = emul_corr_bx;
                stubs_comparison[1].trknmb_emul = emul_trknmb;
                GlobalPoint gp_clct_emul(getGlobalPosition(detid.rawId(), testwg, (*pe).getKeyStrip()));
                stubs_comparison[1].phi_emul = gp_clct_emul.phi();
                bookedclctV_emul[j]  = true;

                //int mingap_trig_pretrig = pretrig_trig_zone*2;
                int num_pretrig = 0;
                for (pretrig = pretrigV_emul.begin(); pretrig != pretrigV_emul.end(); pretrig++){
                  if ((*pretrig).getBX() != (*pe).getBX()) continue;
                  int hsgap = std::abs((*pretrig).getKeyStrip() - (*pe).getKeyStrip());
                  bool samechamber = true;
                  if (stat == 1 and ring == 1)
                    samechamber = (((*pretrig).getKeyStrip() <128 and (*pe).getKeyStrip() <128 ) || ((*pretrig).getKeyStrip() >= 128 and (*pe).getKeyStrip() >=128 ));
                  if (not samechamber)
                    continue;

                  if (hsgap <= pretrig_trig_zone)
                    num_pretrig ++;
                  else
                    continue;
                  if ((*pretrig).getPattern() > stubs_comparison[1].maxpattern_pretrig)
                    stubs_comparison[1].maxpattern_pretrig = (*pretrig).getPattern();
                  if ((*pretrig).getQuality() > stubs_comparison[1].maxquality_pretrig){
                    stubs_comparison[1].quality_pretrig = (*pretrig).getQuality();
                    stubs_comparison[1].key_hs_pretrig = (*pretrig).getKeyStrip();
                    stubs_comparison[1].bend_pretrig = (*pretrig).getBend();
                    stubs_comparison[1].pattern_pretrig = (*pretrig).getPattern();
                    stubs_comparison[1].bx_pretrig = (*pretrig).getBX();
                  }
                }
                stubs_comparison[1].npretrig = num_pretrig;
                break;
              }//if (data_trknmb == emul_trknmb)
            }//loop emul
            if (debug and stubs_comparison[1].key_hs_data != stubs_comparison[1].key_hs_emul)
              LogTrace("CSCTriggerPrimitivesReader")
                <<"CSCDetid "<< detid <<" not matched CLCT from data "<<  (*pd)
                <<" stubs_comparison 1 key_hs_data "<<stubs_comparison[1].key_hs_data <<" key_hs_emul "<< stubs_comparison[1].key_hs_emul << endl;
            //cout <<"CLCT data BX "<< stubs_comparison[1].bx_data <<" emul BX "<< stubs_comparison[1].bx_emul<<" emul BX corrected "<< stubs_comparison[1].bx_corr_emul << endl;
            stub_tree[1]->Fill();
          }//loop data
          for (int k=0; k<nemul; k++){
            if (clctV_emul[i].isValid() == 0) continue;
            if (bookedclctV_emul[k]) continue;
            stubs_comparison[1].init(RUN_, Event_);
            stubs_comparison[1].firstfill = firstfill;
            if (firstfill) firstfill = false;
            stubs_comparison[1].nEvents = eventsAnalyzed;
            stubs_comparison[1].endcap = endc;
            stubs_comparison[1].station = stat;
            stubs_comparison[1].ring = ring;
            stubs_comparison[1].chamber = cham;
            stubs_comparison[1].chambertype = detid.iChamberType();
            stubs_comparison[1].totStubs_data = ndata;
            stubs_comparison[1].totStubs_emul = nemul;
            stubs_comparison[1].totStubs_emul_readout = nemul_readout;
            stubs_comparison[1].trknmb_emul = clctV_emul[i].getTrknmb();
            stubs_comparison[1].nStub_data =-1;
            stubs_comparison[1].has_data = false;
            stubs_comparison[1].nStub_emul = k+1;
            stubs_comparison[1].has_emul = true;
            stubs_comparison[1].quality_emul = clctV_emul[k].getQuality();
            stubs_comparison[1].key_hs_emul = clctV_emul[k].getKeyStrip();
            stubs_comparison[1].bend_emul = clctV_emul[k].getBend();
            stubs_comparison[1].pattern_emul = clctV_emul[k].getPattern();
            stubs_comparison[1].bx_emul = clctV_emul[k].getBX();
            stubs_comparison[1].fullbx_emul = clctV_emul[k].getFullBX();
            if (clctV_emul[k].getKeyStrip()>= 128 and stat == 1 and ring == 1) testwg=5;
            else testwg = 20;
            // Emulator BX NOT Known from  the data.
            GlobalPoint gp_clct_emul(getGlobalPosition(detid.rawId(), testwg, clctV_emul[k].getKeyStrip()));
            stubs_comparison[1].phi_emul = gp_clct_emul.phi();
            bookedclctV_emul[k] = true;
            stub_tree[1]->Fill();
          }
        }
      }
    }
  } // for all chambers
} // compareCLCTs

void CSCTriggerPrimitivesReader::compareLCTs(const CSCCorrelatedLCTDigiCollection* lcts_data,
                                             const CSCCorrelatedLCTDigiCollection* lcts_emul,
                                             const CSCALCTDigiCollection* alcts_data,
                                             const CSCCLCTDigiCollection* clcts_data) {
  // Need ALCT and CLCT digi collections to convert emulator bx into
  // hardware bx.
  // Loop over all chambers in search for correlated LCTs.
  std::vector<CSCCorrelatedLCTDigi>::const_iterator pd, pe;
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= maxRing(stat); ring++) {
        for (int cham = 1; cham <= 36; cham++) {
          // Calculate DetId.  0th layer means whole chamber.
          CSCDetId detid(endc, stat, ring, cham, 0);

          // Skip chambers marked as bad.
          if (checkBadChambers_ && badChambers_->isInBadChamber(detid)) continue;

          std::vector<CSCCorrelatedLCTDigi> lctV_data, lctV_emul;
          std::vector<bool> bookedlctV_emul;
          const auto& drange = lcts_data->get(detid);
          for (auto digiIt = drange.first; digiIt != drange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              lctV_data.push_back(*digiIt);
            }
          }

          const auto& erange = lcts_emul->get(detid);
          for (auto digiIt = erange.first; digiIt != erange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              lctV_emul.push_back(*digiIt);
              bookedlctV_emul.push_back(false);
            }
          }

          int ndata = lctV_data.size();
          int nemul = lctV_emul.size();
          if (ndata == 0 && nemul == 0) continue;
          bool firstfill = true;
          int nemul_readout = 0;
          for (pe = lctV_emul.begin(); pe != lctV_emul.end(); pe++) {
            for (pd = lctV_data.begin(); pd != lctV_data.end(); pd++) {
              int bx_data = pd->getBX();
              int bx_corr = convertBXofLCT((*pe).getBX(), detid,
                                           alcts_data, clcts_data);
              if ((bx_data == bx_corr) or (abs(pe->getKeyWG() - pd->getKeyWG())<=2 ) ){
                nemul_readout++;
                break;
              }

            }
          }

          if (debug or nemul>ndata or (nemul_readout != ndata) or nemul != ndata) {

            LogTrace("CSCTriggerPrimitivesReader") << " LCTs from data "<< ndata <<" LCTs from emul "<< nemul <<" readout "<< nemul_readout;
            ostringstream strstrm;
            strstrm << "\n--- ME" << ((detid.endcap() == 1) ? "+" : "-")
                    << detid.station() << "/" << detid.ring() << "/"
                    << detid.chamber()
                    << " (sector "  << detid.triggerSector()
                    << " trig id. " << detid.triggerCscId() << "):\n";
            strstrm << "  **** " << ndata << " valid data LCTs found:\n";
            for (pd = lctV_data.begin(); pd != lctV_data.end(); pd++) {
              strstrm << "     " << (*pd);
            }
            strstrm << "\n  **** " << nemul << " valid emul LCTs found:\n";
            for (pe = lctV_emul.begin(); pe != lctV_emul.end(); pe++) {
              strstrm << "     " << (*pe);
              strstrm << "    corr BX = "
                      << convertBXofLCT((*pe).getBX(), detid,
                                        alcts_data, clcts_data);
              strstrm << " LCT pattern = " << (*pe).getPattern();
              strstrm << "\n";

            }
            LogTrace("CSCTriggerPrimitivesReader") << strstrm.str();
            //std::cout <<"CompareLCTs "<< strstrm.str()<< std::endl;
          }
          if (nemul_readout > 2){//reduce nemul_readout to 2 by hand
            //cout <<"LCT matching nemul readout is "<< nemul_readout <<", larger than 2. reduce it to 2 by hand"<<endl;
            nemul_readout = 2;
          }

          //
          //Data, add HS quality later
          int perEv_nStub_data = 0;
          int perEv_nStub_emul = 0;
          perStub[4].init(RUN_, Event_);
          perStub[5].init(RUN_, Event_);
          for (pd = lctV_data.begin(); pd != lctV_data.end(); pd++){
            perEv_nStub_data++;
          }
          perStub[4].t_endcap = endc;
          perStub[4].t_station = stat;
          perStub[4].t_chamber = cham;
          perStub[4].t_ring = ring;
          perStub[4].t_EventNumberAnalyzed = eventsAnalyzed;
          perStub[4].t_nStubs              = ndata;
          perStub[4].t_nStubs_readout              = ndata;
          event_tree[4]->Fill();
          //Emul
          for (pe = lctV_emul.begin(); pe != lctV_emul.end(); pe++){
            perEv_nStub_emul++;
          }
          perStub[5].t_endcap = endc;
          perStub[5].t_station = stat;
          perStub[5].t_chamber = cham;
          perStub[5].t_ring = ring;
          perStub[5].t_EventNumberAnalyzed = eventsAnalyzed;
          perStub[5].t_nStubs              = nemul;
          perStub[5].t_nStubs_readout      = nemul_readout;
          event_tree[5]->Fill();

          int csctype = getCSCType(detid);
          hLctCompFoundCsc[endc-1][csctype]->Fill(cham);
          int ix = chamberIX(detid);
          int ix2 = chamberIXi(detid);
          if(detid.station()>1 && detid.ring()==1) {
            hLCTCompFound2x->Fill(ix,detid.chamber()*2);
          }
          else {
            hLCTCompFound2->Fill(ix,detid.chamber());
          }
          hLCTCompFound2i->Fill(ix2,detid.chamber());
          if (ndata != nemul) {
            LogTrace("CSCTriggerPrimitivesReader")
              //cerr
              << "   +++ Different numbers of LCTs found in ME"
              << ((endc == 1) ? "+" : "-") << stat << "/"
              << ring << "/" << cham
              << ": data = " << ndata << " emulator = " << nemul << " +++\n";
          }
          else {
            hLctCompSameNCsc[endc-1][csctype]->Fill(cham);
            if(detid.station()>1 && detid.ring()==1) {
              hLCTCompSameN2x->Fill(ix,detid.chamber()*2);
            }
            else {
              hLCTCompSameN2->Fill(ix,detid.chamber());
            }
            hLCTCompSameN2i->Fill(ix2,detid.chamber());
          }
          int i =-1;
          for (pd = lctV_data.begin(); pd != lctV_data.end(); pd++) {
            i++;
            if ((*pd).isValid() == 0) continue;
            int data_trknmb    = (*pd).getTrknmb();
            int data_quality   = (*pd).getQuality();
            int data_wiregroup = (*pd).getKeyWG();
            int data_keystrip  = (*pd).getStrip();
            int data_pattern   = (*pd).getCLCTPattern();
            int data_striptype = (*pd).getStripType();
            int data_bend      = (*pd).getBend();
            int data_bx        = (*pd).getBX();

            stubs_comparison[2].init(RUN_, Event_);
            stubs_comparison[2].firstfill = firstfill;
            if (firstfill) firstfill = false;
            stubs_comparison[2].nEvents = eventsAnalyzed;
            stubs_comparison[2].endcap = endc;
            stubs_comparison[2].station = stat;
            stubs_comparison[2].ring = ring;
            stubs_comparison[2].chamber = cham;
            stubs_comparison[2].chambertype = detid.iChamberType();
            stubs_comparison[2].totStubs_data = ndata;
            stubs_comparison[2].totStubs_emul = nemul;
            stubs_comparison[2].totStubs_emul_readout = nemul_readout;
            stubs_comparison[2].nStub_data = i+1;
            stubs_comparison[2].has_data = true;
            stubs_comparison[2].quality_data = (*pd).getQuality();
            stubs_comparison[2].key_WG_data = (*pd).getKeyWG();
            stubs_comparison[2].key_hs_data = (*pd).getStrip();
            //cout <<" CSC detid "<< detid <<" LCT in data "<< (*pd) <<" do doesALCTCrossCLCT "<< endl;
            stubs_comparison[2].WGcrossHS_data = doesALCTCrossCLCT(detid,  (*pd).getKeyWG(),  (*pd).getStrip());
            stubs_comparison[2].bend_data = (*pd).getBend();
            stubs_comparison[2].pattern_data = (*pd).getCLCTPattern();
            stubs_comparison[2].bx_data = (*pd).getBX();
            stubs_comparison[2].trknmb_data = data_trknmb;
            GlobalPoint gp_lct_data(getGlobalPosition(detid.rawId(), (*pd).getKeyWG(), (*pd).getStrip()));
            stubs_comparison[2].eta_data = gp_lct_data.eta();
            stubs_comparison[2].phi_data = gp_lct_data.phi();
            int j=-1;
            for (pe = lctV_emul.begin(); pe != lctV_emul.end(); pe++) {
              j++;
              if ((*pe).isValid() == 0) continue;
              if (bookedlctV_emul[j]) continue; //used alct
              int emul_trknmb    = (*pe).getTrknmb();
              int emul_quality   = (*pe).getQuality();
              int emul_wiregroup = (*pe).getKeyWG();
              int emul_keystrip  = (*pe).getStrip();
              int emul_pattern   = (*pe).getCLCTPattern();
              int emul_striptype = (*pe).getStripType();
              int emul_bend      = (*pe).getBend();
              int emul_bx        = (*pe).getBX();
              if (abs(data_keystrip - emul_keystrip) <= 2 || abs(data_wiregroup - emul_wiregroup) <= 2){
                // Convert emulator BX into hardware BX using full 12-bit
                // BX words in ALCT and CLCT digi collections.
                int emul_corr_bx = convertBXofLCT(emul_bx, detid,
                                                  alcts_data, clcts_data);
                //std::cout <<"LCT data_bx "<< data_bx <<" emul_corr_bx "<< emul_corr_bx << std::endl;
                if (dataIsAnotherMC_)
                  emul_corr_bx = (emul_bx & 0x01);

                if (ndata == nemul) {
                  hLctCompTotalCsc[endc-1][csctype]->Fill(cham);
                  if(detid.station()>1 && detid.ring()==1) {
                    hLCTCompTotal2x->Fill(ix,detid.chamber()*2);
                  }
                  else {
                    hLCTCompTotal2->Fill(ix,detid.chamber());
                  }
                  hLCTCompTotal2i->Fill(ix2,detid.chamber());
                }
                if (data_quality   == emul_quality   &&
                    data_wiregroup == emul_wiregroup &&
                    data_keystrip  == emul_keystrip  &&
                    data_pattern   == emul_pattern   &&
                    data_striptype == emul_striptype &&
                    data_bend      == emul_bend      ){
                  if (ndata == nemul) {
                    hLctCompMatchCsc[endc-1][csctype]->Fill(cham);
                    if(detid.station()>1 && detid.ring()==1) {
                      hLCTCompMatch2x->Fill(ix,detid.chamber()*2);
                    }
                    else {
                      hLCTCompMatch2->Fill(ix,detid.chamber());
                    }
                    hLCTCompMatch2i->Fill(ix2,detid.chamber());
                  }
                  if (debug) LogTrace("CSCTriggerPrimitivesReader")
                               << "       Identical LCTs #" << data_trknmb;
                }
                else {
                  LogTrace("CSCTriggerPrimitivesReader")
                    << "       Different LCTs #" << data_trknmb << " in ME"
                    << ((endc == 1) ? "+" : "-") << stat << "/"
                    << ring << "/" << cham <<" data_lct bx "<< data_bx<< " emul bx "<< emul_corr_bx;
                }
                stubs_comparison[2].nStub_emul = j+1;
                stubs_comparison[2].has_emul = true;
                stubs_comparison[2].quality_emul = (*pe).getQuality();
                stubs_comparison[2].key_WG_emul = (*pe).getKeyWG();
                stubs_comparison[2].key_hs_emul = (*pe).getStrip();
                stubs_comparison[2].bend_emul = (*pe).getBend();
                stubs_comparison[2].pattern_emul = (*pe).getCLCTPattern();
                stubs_comparison[2].bx_emul = (*pe).getBX();
                //cout <<" CSC detid "<< detid <<" LCT in emul "<< (*pe) <<" do doesALCTCrossCLCT "<< endl;
                stubs_comparison[2].WGcrossHS_emul = doesALCTCrossCLCT(detid,  (*pe).getKeyWG(),  (*pe).getStrip());
                stubs_comparison[2].bx_corr_emul = emul_corr_bx;
                stubs_comparison[2].trknmb_emul = emul_trknmb;
                GlobalPoint gp_lct_emul(getGlobalPosition(detid.rawId(), (*pe).getKeyWG(), (*pe).getStrip()));
                stubs_comparison[2].eta_emul = gp_lct_emul.eta();
                stubs_comparison[2].phi_emul = gp_lct_emul.phi();
                bookedlctV_emul[j]  = true;
                break;
              }//if (data_trknmb == emul_trknmb)
            }//loop emul
            if (debug and stubs_comparison[2].key_hs_data != stubs_comparison[2].key_hs_emul)
              LogTrace("CSCTriggerPrimitivesReader")
                <<"CSCDetid "<< detid <<" not matched LCT from Data "<< (*pd)
                <<" stubs_comparison 2 key_hs_data "<<stubs_comparison[2].key_hs_data <<" key_hs_emul "<< stubs_comparison[2].key_hs_emul << endl;
            stub_tree[2]->Fill();
          }//loop data
          for (int k=0; k<nemul; k++){
            if (bookedlctV_emul[k]) continue;
            if (lctV_emul[i].isValid() == 0) continue;
            stubs_comparison[2].init(RUN_, Event_);
            stubs_comparison[2].nEvents = eventsAnalyzed;
            stubs_comparison[2].endcap = endc;
            stubs_comparison[2].station = stat;
            stubs_comparison[2].ring = ring;
            stubs_comparison[2].chamber = cham;
            stubs_comparison[2].chambertype = detid.iChamberType();
            stubs_comparison[2].totStubs_data = ndata;
            stubs_comparison[2].totStubs_emul = nemul;
            stubs_comparison[2].totStubs_emul_readout = nemul_readout;
            stubs_comparison[2].trknmb_emul = lctV_emul[i].getTrknmb();
            stubs_comparison[2].nStub_data =-1;
            stubs_comparison[2].has_data = false;
            stubs_comparison[2].nStub_emul = k+1;
            stubs_comparison[2].has_emul = true;
            stubs_comparison[2].WGcrossHS_emul = doesALCTCrossCLCT(detid,  lctV_emul[k].getKeyWG(),  lctV_emul[k].getStrip());
            stubs_comparison[2].quality_emul = lctV_emul[k].getQuality();
            stubs_comparison[2].key_WG_emul = lctV_emul[k].getKeyWG();
            stubs_comparison[2].key_hs_emul = lctV_emul[k].getStrip();
            stubs_comparison[2].bend_emul = lctV_emul[k].getBend();
            stubs_comparison[2].pattern_emul = lctV_emul[k].getCLCTPattern();
            stubs_comparison[2].bx_emul = lctV_emul[k].getBX();
            //stubs_comparison[2].fullbx_emul = lctV_emul[k].getFullBX();
            // Emulator BX NOT Known from  the data.
            GlobalPoint gp_lct_emul(getGlobalPosition(detid.rawId(), lctV_emul[k].getKeyWG(), lctV_emul[k].getStrip()));
            stubs_comparison[2].eta_emul = gp_lct_emul.eta();
            stubs_comparison[2].phi_emul = gp_lct_emul.phi();
            bookedlctV_emul[k] = true;
            stub_tree[2]->Fill();

          }
        }
      }
    }
  }
}


void CSCTriggerPrimitivesReader::compareMPCLCTs(
                                                const CSCCorrelatedLCTDigiCollection* lcts_data,
                                                const CSCCorrelatedLCTDigiCollection* lcts_emul,
                                                const CSCALCTDigiCollection* alcts_data,
                                                const CSCCLCTDigiCollection* clcts_data) {

  // Need ALCT and CLCT digi collections to convert emulator bx into
  // hardware bx.
  // Loop over all chambers in search for correlated LCTs.
  std::vector<CSCCorrelatedLCTDigi>::const_iterator pd, pe;
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= maxRing(stat); ring++) {
        for (int cham = 1; cham <= 36; cham++) {
          // Calculate DetId.  0th layer means whole chamber.
          CSCDetId detid(endc, stat, ring, cham, 0);

          // Skip chambers marked as bad.
          if (checkBadChambers_ && badChambers_->isInBadChamber(detid)) continue;

          std::vector<CSCCorrelatedLCTDigi> lctV_data, lctV_emul;
          std::vector<bool> bookedlctV_emul;
          const auto& drange = lcts_data->get(detid);
          for (auto digiIt = drange.first; digiIt != drange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              lctV_data.push_back(*digiIt);
            }
          }

          const auto& erange = lcts_emul->get(detid);
          for (auto digiIt = erange.first; digiIt != erange.second; digiIt++) {
            if ((*digiIt).isValid()) {
              lctV_emul.push_back(*digiIt);
              bookedlctV_emul.push_back(false);
            }
          }

          int ndata = lctV_data.size();
          int nemul = lctV_emul.size();
          if (ndata == 0 && nemul == 0) continue;
          bool firstfill = true;

          if (debug) {
            ostringstream strstrm;
            strstrm << "\n--- ME" << ((detid.endcap() == 1) ? "+" : "-")
                    << detid.station() << "/" << detid.ring() << "/"
                    << detid.chamber()
                    << " (sector "  << detid.triggerSector()
                    << " trig id. " << detid.triggerCscId() << "):\n";
            strstrm << "  **** " << ndata << " valid data LCTs found:\n";
            for (pd = lctV_data.begin(); pd != lctV_data.end(); pd++) {
              strstrm << "     " << (*pd);
            }
            strstrm << "\n  **** " << nemul << " valid emul LCTs found:\n";
            for (pe = lctV_emul.begin(); pe != lctV_emul.end(); pe++) {
              strstrm << "     " << (*pe);
              strstrm << "    corr BX = "
                      << convertBXofLCT((*pe).getBX(), detid,
                                        alcts_data, clcts_data);
              strstrm << " LCT pattern = " << (*pe).getPattern();
              strstrm << "\n";
            }
            LogTrace("CSCTriggerPrimitivesReader") << strstrm.str();
            //std::cout <<"CompareLCTs "<< strstrm.str()<< std::endl;
          }

          //int ix = chamberIX(detid);
          //int ix2 = chamberIXi(detid);
          if (ndata != nemul) {
            LogTrace("CSCTriggerPrimitivesReader")
              << "   +++ Different numbers of MPC LCTs found in ME"
              << ((endc == 1) ? "+" : "-") << stat << "/"
              << ring << "/" << cham
              << ": data = " << ndata << " emulator = " << nemul << " +++\n";
          }
          /*else {
            hLctCompSameNCsc[endc-1][csctype]->Fill(cham);
            if(detid.station()>1 && detid.ring()==1) {
            hLCTCompSameN2x->Fill(ix,detid.chamber()*2);
            }
            else {
            hLCTCompSameN2->Fill(ix,detid.chamber());
            }
            hLCTCompSameN2i->Fill(ix2,detid.chamber());
            }*/
          int i =-1;
          for (pd = lctV_data.begin(); pd != lctV_data.end(); pd++) {
            i++;
            if ((*pd).isValid() == 0) continue;
            int data_trknmb    = (*pd).getTrknmb();
            int data_quality   = (*pd).getQuality();
            int data_wiregroup = (*pd).getKeyWG();
            int data_keystrip  = (*pd).getStrip();
            int data_pattern   = (*pd).getCLCTPattern();
            int data_striptype = (*pd).getStripType();
            int data_bend      = (*pd).getBend();
            int data_bx        = (*pd).getBX();

            stubs_comparison[3].init(RUN_, Event_);
            stubs_comparison[3].firstfill = firstfill;
            if (firstfill) firstfill = false;
            stubs_comparison[3].nEvents = eventsAnalyzed;
            stubs_comparison[3].endcap = endc;
            stubs_comparison[3].station = stat;
            stubs_comparison[3].ring = ring;
            stubs_comparison[3].chamber = cham;
            stubs_comparison[3].chambertype = detid.iChamberType();
            stubs_comparison[3].totStubs_data = ndata;
            stubs_comparison[3].totStubs_emul = nemul;
            stubs_comparison[3].nStub_data = i+1;
            stubs_comparison[3].has_data = true;
            stubs_comparison[3].WGcrossHS_data = doesALCTCrossCLCT(detid,  (*pd).getKeyWG(),  (*pd).getStrip());
            stubs_comparison[3].quality_data = (*pd).getQuality();
            stubs_comparison[3].key_WG_data = (*pd).getKeyWG();
            stubs_comparison[3].key_hs_data = (*pd).getStrip();
            stubs_comparison[3].bend_data = (*pd).getBend();
            stubs_comparison[3].pattern_data = (*pd).getCLCTPattern();
            stubs_comparison[3].bx_data = (*pd).getBX();
            stubs_comparison[3].trknmb_data = data_trknmb;
            GlobalPoint gp_lct_data(getGlobalPosition(detid.rawId(), (*pd).getKeyWG(), (*pd).getStrip()));
            stubs_comparison[3].eta_data = gp_lct_data.eta();
            stubs_comparison[3].phi_data = gp_lct_data.phi();
            int j=-1;
            for (pe = lctV_emul.begin(); pe != lctV_emul.end(); pe++) {
              j++;
              if ((*pe).isValid() == 0) continue;
              int emul_trknmb    = (*pe).getTrknmb();
              int emul_quality   = (*pe).getQuality();
              int emul_wiregroup = (*pe).getKeyWG();
              int emul_keystrip  = (*pe).getStrip();
              int emul_pattern   = (*pe).getCLCTPattern();
              int emul_striptype = (*pe).getStripType();
              int emul_bend      = (*pe).getBend();
              int emul_bx        = (*pe).getBX();
              if (data_trknmb == emul_trknmb) {
                // Convert emulator BX into hardware BX using full 12-bit
                // BX words in ALCT and CLCT digi collections.
                int emul_corr_bx = convertBXofLCT(emul_bx, detid,
                                                  alcts_data, clcts_data);
                //std::cout <<"LCT data_bx "<< data_bx <<" emul_corr_bx "<< emul_corr_bx << std::endl;
                if (dataIsAnotherMC_)
                  emul_corr_bx = (emul_bx & 0x01);

                /*if (ndata == nemul) {
                  hLctCompTotalCsc[endc-1][csctype]->Fill(cham);
                  if(detid.station()>1 && detid.ring()==1) {
                  hLCTCompTotal2x->Fill(ix,detid.chamber()*2);
                  }
                  else {
                  hLCTCompTotal2->Fill(ix,detid.chamber());
                  }
                  hLCTCompTotal2i->Fill(ix2,detid.chamber());
                  }*/
                if (data_quality   == emul_quality   &&
                    data_wiregroup == emul_wiregroup &&
                    data_keystrip  == emul_keystrip  &&
                    data_pattern   == emul_pattern   &&
                    data_striptype == emul_striptype &&
                    data_bend      == emul_bend      ){
                  //data_bx        == emul_corr_bx) {
                  /*if (ndata == nemul) {
                    hLctCompMatchCsc[endc-1][csctype]->Fill(cham);
                    if(detid.station()>1 && detid.ring()==1) {
                    hLCTCompMatch2x->Fill(ix,detid.chamber()*2);
                    }
                    else {
                    hLCTCompMatch2->Fill(ix,detid.chamber());
                    }
                    hLCTCompMatch2i->Fill(ix2,detid.chamber());
                    }*/
                  if (debug) LogTrace("CSCTriggerPrimitivesReader")
                               << "       Identical LCTs #" << data_trknmb;
                  stubs_comparison[3].nStub_emul = j+1;
                  stubs_comparison[3].has_emul = true;
                  stubs_comparison[3].WGcrossHS_emul = doesALCTCrossCLCT(detid,  (*pe).getKeyWG(),  (*pe).getStrip());
                  stubs_comparison[3].quality_emul = (*pe).getQuality();
                  stubs_comparison[3].key_WG_emul = (*pe).getKeyWG();
                  stubs_comparison[3].key_hs_emul = (*pe).getStrip();
                  stubs_comparison[3].bend_emul = (*pe).getBend();
                  stubs_comparison[3].pattern_emul = (*pe).getCLCTPattern();
                  stubs_comparison[3].bx_emul = (*pe).getBX();
                  stubs_comparison[3].bx_corr_emul = emul_corr_bx;
                  stubs_comparison[3].trknmb_emul = emul_trknmb;
                  GlobalPoint gp_lct_emul(getGlobalPosition(detid.rawId(), (*pe).getKeyWG(), (*pe).getStrip()));
                  stubs_comparison[3].eta_emul = gp_lct_emul.eta();
                  stubs_comparison[3].phi_emul = gp_lct_emul.phi();
                  bookedlctV_emul[j]  = true;
                  break;
                }
                else {
                  LogTrace("CSCTriggerPrimitivesReader")
                    //cerr
                    << "       Different LCTs #" << data_trknmb << " in ME"
                    << ((endc == 1) ? "+" : "-") << stat << "/"
                    << ring << "/" << cham <<" data_lct bx "<< data_bx<< " emul bx "<< emul_corr_bx;
                }
              }
            }//loop emul
            if (debug and stubs_comparison[3].key_hs_data != stubs_comparison[3].key_hs_emul)
              LogTrace("CSCTriggerPrimitivesReader")
                <<"stubs_comparison 2 key_hs_data "<<stubs_comparison[3].key_hs_data <<" key_hs_emul "<< stubs_comparison[3].key_hs_emul;
            stub_tree[3]->Fill();
          }//loop data

          for (int k=0; k<nemul; k++){
            if (bookedlctV_emul[k]) continue;
            stubs_comparison[3].init(RUN_, Event_);
            stubs_comparison[3].firstfill = firstfill;
            if (firstfill) firstfill = false;
            stubs_comparison[3].nEvents = eventsAnalyzed;
            stubs_comparison[3].endcap = endc;
            stubs_comparison[3].station = stat;
            stubs_comparison[3].ring = ring;
            stubs_comparison[3].chamber = cham;
            stubs_comparison[3].chambertype = detid.iChamberType();
            stubs_comparison[3].totStubs_data = ndata;
            stubs_comparison[3].totStubs_emul = nemul;
            stubs_comparison[3].trknmb_emul = lctV_emul[i].getTrknmb();
            stubs_comparison[3].nStub_data =-1;
            stubs_comparison[3].has_data = false;
            stubs_comparison[3].nStub_emul = k+1;
            stubs_comparison[3].has_emul = true;
            stubs_comparison[3].WGcrossHS_emul = doesALCTCrossCLCT(detid, lctV_emul[k].getKeyWG(),  lctV_emul[k].getStrip());
            stubs_comparison[3].quality_emul = lctV_emul[k].getQuality();
            stubs_comparison[3].key_WG_emul = lctV_emul[k].getKeyWG();
            stubs_comparison[3].key_hs_emul = lctV_emul[k].getStrip();
            stubs_comparison[3].bend_emul = lctV_emul[k].getBend();
            stubs_comparison[3].pattern_emul = lctV_emul[k].getCLCTPattern();
            stubs_comparison[3].bx_emul = lctV_emul[k].getBX();
            //stubs_comparison[2].fullbx_emul = lctV_emul[k].getFullBX();
            // Emulator BX NOT Known from  the data.
            GlobalPoint gp_lct_emul(getGlobalPosition(detid.rawId(), lctV_emul[k].getKeyWG(), lctV_emul[k].getStrip()));
            stubs_comparison[3].eta_emul = gp_lct_emul.eta();
            stubs_comparison[3].phi_emul = gp_lct_emul.phi();
            bookedlctV_emul[k] = true;
            stub_tree[3]->Fill();

          }
        }//end loop of chambers
      }
    }
  }
}




int CSCTriggerPrimitivesReader::convertBXofLCT(
                                               const int emul_bx, const CSCDetId& detid,
                                               const CSCALCTDigiCollection* alcts_data,
                                               const CSCCLCTDigiCollection* clcts_data) {
  int full_anode_bx = -999;
  //int full_cathode_bx = -999;
  int lct_bx = -999;
  int tbin_anode_offset = 5; // 2007, run 14419.

  // Extract full 12-bit anode BX word from ALCT collections.
  const auto& arange = alcts_data->get(detid);
  for (auto digiIt = arange.first; digiIt != arange.second; digiIt++) {
    if ((*digiIt).isValid()) {
      full_anode_bx = (*digiIt).getFullBX();
      break;
    }
  }

  // Extract full 12-bit cathode BX word from CLCT collections.
  const auto& crange = clcts_data->get(detid);
  for (auto digiIt = crange.first; digiIt != crange.second; digiIt++) {
    if ((*digiIt).isValid()) {
      //full_cathode_bx = (*digiIt).getFullBX();
      break;
    }
  }

  // Use these 12-bit BX's to convert emulator BX into hardware BX.
  if (full_anode_bx == -999) {
    // What to do???
    edm::LogWarning("L1CSCTPEmulatorWrongInput")
      << "+++ Warning in convertBXofLCT(): full anode BX is not available!"
      << " +++\n";
  }
  else {
    // LCT BX has two bits: the least-significant bit is the LSB of ALCT BX;
    // the MSB is 1/0 depending on whether the 12-bit full cathode BX is 0
    // or not.
    lct_bx = (full_anode_bx + emul_bx - tbin_anode_offset) & 0x01;
    // SV, 12/Jun/08: it looks like this bit is never set - docu must be
    // wrong.
    //lct_bx = lct_bx | ((full_cathode_bx == 0) << 1);
  }
  //std::cout <<"convertBXofLCT old emul_bx "<< emul_bx <<" new lct bx "<< lct_bx <<" ful_anode_bx "<< full_anode_bx << std::endl;
  return lct_bx;
}


void CSCTriggerPrimitivesReader::HotWires(const edm::Event& iEvent) {
  if (!bookedHotWireHistos) bookHotWireHistos();
  edm::Handle<CSCWireDigiCollection> wires;
  //  iEvent.getByLabel(wireDigiProducer_.label(), wireDigiProducer_.instance(), wires);
  iEvent.getByToken(wireDigi_token_, wires);

  int serial_old=-1;
  for (auto dWDiter=wires->begin(); dWDiter!=wires->end(); dWDiter++) {
    CSCDetId id = (CSCDetId)(*dWDiter).first;
    int serial = chamberSerial(id)-1;
    //     printf("serial %i\n",serial);
    std::vector<CSCWireDigi>::const_iterator wireIter = (*dWDiter).second.first;
    std::vector<CSCWireDigi>::const_iterator lWire = (*dWDiter).second.second;
    bool has_layer=false;
    for( ; wireIter != lWire; ++wireIter) {
      has_layer=true;
      int i_layer= id.layer()-1;
      int i_wire = wireIter->getWireGroup()-1;
      int nbins = wireIter->getTimeBinsOn().size();
      int serial2=serial*(6*112)+i_layer*112+i_wire;
      /*
        printf("endcap %i, station %i, ring %i, chamber %i, serial %i, serial2 %i, layer %i, wg %i\n",
        id.endcap(), id.station(), id.ring(), id.chamber(), serial, serial2, i_layer, i_wire);
      */
      hHotWire1->Fill(serial2,nbins);
      /*
        if(id.station()==1) {
        printf("ring %i, chamber type %i, wg %i\n",id.ring(),id.iChamberType(),i_wire);
        }
      */
    }
    if(serial_old!=serial && has_layer) {
      //       nHotCham[serial]++;
      hHotCham1->Fill(serial);
      //	printf("serial %i filled\n",serial);
      serial_old=serial;
    }
  }
}

void CSCTriggerPrimitivesReader::MCStudies(const edm::Event& ev,
                                           const CSCALCTDigiCollection* alcts,
                                           const CSCCLCTDigiCollection* clcts) {
  // MC particles, if any.
  //edm::Handle<edm::HepMCProduct> mcp;
  //ev.getByLabel("source", mcp);
  //ev.getByType(mcp);
  vector<edm::Handle<edm::HepMCProduct> > allhepmcp;
  // Use "getManyByType" to be able to check the existence of MC info.
  ev.getManyByType(allhepmcp);

  //cout << "HepMC info: " << allhepmcp.size() << endl;
  if (allhepmcp.size() > 0) {
    const HepMC::GenEvent& mc = allhepmcp[0]->getHepMCData();
    int i = 0;
    for (HepMC::GenEvent::particle_const_iterator p = mc.particles_begin();
         p != mc.particles_end(); ++p) {
      int id = (*p)->pdg_id();
      double phitmp = (*p)->momentum().phi();
      if (phitmp < 0) phitmp += 2.*M_PI;
      if (debug) LogDebug("CSCTriggerPrimitivesReader")
                   << "MC part #" << ++i << ": id = "  << id
                   << ", status = " << (*p)->status()
                   << "\n   pX = " << (*p)->momentum().x()
                   << ", pY = " << (*p)->momentum().y()
                   << ", pT = " << (*p)->momentum().perp() << " GeV"
                   << ", p =  " << (*p)->momentum().rho()  << " GeV"
                   << "\n   eta = " << (*p)->momentum().pseudoRapidity()
                   << ", phi = " << phitmp << " (" << phitmp*180./M_PI << " deg)";
    }

    // If hepMC info is there, try to get wire and comparator digis,
    // and SimHits.
    edm::Handle<CSCWireDigiCollection>       wireDigis;
    edm::Handle<CSCComparatorDigiCollection> compDigis;
    edm::Handle<edm::PSimHitContainer>       simHitsH;
    //    ev.getByLabel(wireDigiProducer_.label(), wireDigiProducer_.instance(),
    //		  wireDigis);
    //    ev.getByLabel(compDigiProducer_.label(), compDigiProducer_.instance(),
    //		  compDigis);
    //    ev.getByLabel(simHitProducer_.label(), simHitProducer_.instance(),
    //		  simHits);
    ev.getByToken(wireDigi_token_, wireDigis);
    ev.getByToken(compDigi_token_, compDigis);
    ev.getByToken(simHit_token_, simHitsH);
    const edm::PSimHitContainer& simHits = *simHitsH.product();

    if (!wireDigis.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of wire digis with label"
        << wireDigiProducer_.label()
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }
    if (!compDigis.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of comparator digis with label"
        << compDigiProducer_.label()
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }
    if (!simHitsH.isValid()) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
        << "+++ Warning: Collection of SimHits with label"
        << simHitProducer_.label()
        << " requested, but not found in the event... Skipping the rest +++\n";
      return;
    }


    if (debug) LogTrace("CSCTriggerPrimitivesReader")
                 << "   #CSC SimHits: " << simHits.size();

    // MC-based resolution studies.
    calcResolution(alcts, clcts, wireDigis.product(), compDigis.product(),
                   simHits);

    // MC-based efficiency studies.
    calcEfficiency(alcts, clcts, simHits);
  }
}

void CSCTriggerPrimitivesReader::calcResolution(const CSCALCTDigiCollection* alcts, const CSCCLCTDigiCollection* clcts,
                                                const CSCWireDigiCollection* wiredc,
                                                const CSCComparatorDigiCollection* compdc,
                                                const edm::PSimHitContainer& allSimHits) {

  // Book histos when called for the first time.
  if (!bookedResolHistos) bookResolHistos();

  // ALCT resolution
  CSCAnodeLCTAnalyzer alct_analyzer;
  alct_analyzer.setGeometry(geom_);

  for (auto adetUnitIt = alcts->begin(); adetUnitIt != alcts->end(); adetUnitIt++) {
    const CSCDetId& id = (*adetUnitIt).first;
    if (checkBadChambers_ && badChambers_->isInBadChamber(id)) continue;
    const auto& range = (*adetUnitIt).second;
    for (auto digiIt = range.first;
         digiIt != range.second; digiIt++) {

      bool alct_valid = (*digiIt).isValid();
      if (alct_valid) {
        vector<CSCAnodeLayerInfo> alctInfo =
          alct_analyzer.getSimInfo(*digiIt, id, wiredc, &allSimHits);

        double hitPhi = -999.0, hitEta = -999.0;
        int hitWG = alct_analyzer.nearestWG(alctInfo, hitPhi, hitEta);
        if (hitWG >= 0.) {
          // Key wire group and key layer id.
          int wiregroup = (*digiIt).getKeyWG();

          CSCDetId layerId(id.endcap(), id.station(), id.ring(),
                           id.chamber(), CSCConstants::KEY_ALCT_LAYER);
          int endc    = id.endcap();
          int stat    = id.station();
          int csctype = getCSCType(id);

          double alctEta  = alct_analyzer.getWGEta(layerId, wiregroup);
          double deltaEta = alctEta - hitEta;
          hResolDeltaEta->Fill(deltaEta);

          double deltaWG = wiregroup - hitWG;
          if (debug) LogTrace("CSCTriggerPrimitivesReader")
                       << "WG: MC = " << hitWG << " rec = " << wiregroup
                       << " delta = " << deltaWG;
          hResolDeltaWG->Fill(deltaWG);

          hEtaRecVsSim->Fill(fabs(hitEta), fabs(alctEta));
          hEtaDiffCsc[csctype][0]->Fill(deltaEta);
          hEtaDiffCsc[csctype][endc]->Fill(deltaEta);
          hAlctVsEta[stat-1]->Fill(fabs(alctEta));
          hEtaDiffVsEta[stat-1]->Fill(fabs(alctEta), fabs(deltaEta));
          hEtaDiffVsWireCsc[csctype]->Fill(wiregroup, deltaEta);
        }
        // should I comment out this "else"?
        //else {
        //  edm::LogWarning("L1CSCTPEmulatorWrongInput")
        //    << "+++ Warning in calcResolution(): no matched SimHit"
        //    << " found! +++\n";
        //}
      }
    }
  }

  // CLCT resolution
  static const int key_layer = CSCConstants::KEY_CLCT_LAYER;
  CSCCathodeLCTAnalyzer clct_analyzer;
  clct_analyzer.setGeometry(geom_);

  for (auto cdetUnitIt = clcts->begin(); cdetUnitIt != clcts->end(); cdetUnitIt++) {
    const CSCDetId& id = (*cdetUnitIt).first;
    if (checkBadChambers_ && badChambers_->isInBadChamber(id)) continue;
    const auto& range = (*cdetUnitIt).second;
    for (auto digiIt = range.first;
         digiIt != range.second; digiIt++) {

      bool clct_valid = (*digiIt).isValid();
      if (clct_valid) {
        vector<CSCCathodeLayerInfo> clctInfo =
          clct_analyzer.getSimInfo(*digiIt, id, compdc, &allSimHits);

        double hitPhi = -999.0, hitEta = -999.0, deltaStrip = -999.0;
        int hitHS = clct_analyzer.nearestHS(clctInfo, hitPhi, hitEta);
        if (hitHS >= 0.) {
          // Key strip and key layer id.
          int halfstrip = (*digiIt).getKeyStrip();
          int strip     = halfstrip/2;
          int distrip   = halfstrip/4;
          int stripType = (*digiIt).getStripType();

          CSCDetId layerId(id.endcap(), id.station(), id.ring(),
                           id.chamber(), key_layer);
          int endc    = id.endcap();
          int stat    = id.station();
          int csctype = getCSCType(id);

          // 'float strip' is in the units of 'strip', i.e., angular
          // widths of each strip. The counting is from 0.0 at the extreme
          // edge of the 'first' strip at one edge of the detector.
          float fstrip = -999.;
          if (stripType == 0) { // di-strip CLCT
            fstrip = strip + 1.;
          }
          else {                // half-strip CLCT
            fstrip = strip + 0.5*(halfstrip%2) + 0.25;
          }
          double clctPhi = clct_analyzer.getStripPhi(layerId, fstrip);
          double deltaPhi = clctPhi - hitPhi;
          if      (deltaPhi < -M_PI) deltaPhi += 2.*M_PI;
          else if (deltaPhi >  M_PI) deltaPhi -= 2.*M_PI;
          deltaPhi *= 1000; // in mrad
          if      (hitPhi  < 0) hitPhi  += 2.*M_PI;
          if      (clctPhi < 0) clctPhi += 2.*M_PI;
          if (debug) LogTrace("CSCTriggerPrimitivesReader")
                       << " clctPhi = " << clctPhi << " hitPhi = " << hitPhi
                       << " deltaPhi = " << deltaPhi;

          hResolDeltaPhi->Fill(deltaPhi);
          if (stripType == 0) { // di-strip CLCT
            deltaStrip = distrip - hitHS/4;
            hResolDeltaDS->Fill(deltaStrip);
            hResolDeltaPhiDS->Fill(deltaPhi);
            hPhiDiffVsStripCsc[csctype][0]->Fill(distrip,   deltaPhi);
          }
          else {                // half-strip CLCT
            deltaStrip = halfstrip - hitHS;
            hResolDeltaHS->Fill(deltaStrip);
            hResolDeltaPhiHS->Fill(deltaPhi);
            hPhiDiffVsStripCsc[csctype][1]->Fill(halfstrip, deltaPhi);
          }
          if (debug) LogTrace("CSCTriggerPrimitivesReader")
                       << "Half-strip: MC = " << hitHS << " rec = " << halfstrip
                       << " pattern type = " << stripType << " delta = " << deltaStrip;

          hPhiRecVsSim->Fill(hitPhi, clctPhi);
          hPhiDiffCsc[csctype][0]->Fill(deltaPhi);
          hPhiDiffCsc[csctype][endc]->Fill(deltaPhi);
          hPhiDiffCsc[csctype][stripType+3]->Fill(deltaPhi);
          hClctVsPhi[stat-1]->Fill(clctPhi);
          hPhiDiffVsPhi[stat-1]->Fill(clctPhi, fabs(deltaPhi));

          // Histograms to check phi offsets for various pattern types
          if (stripType == 1) { // half-strips
            double hsperrad = getHsPerRad(csctype); // halfstrips-per-radian
            if((endc == 1 && (stat == 1 || stat == 2)) ||
               (endc == 2 && (stat == 3 || stat == 4))) {
              hPhiDiffPattern[(*digiIt).getPattern()]->Fill(deltaPhi/1000*hsperrad);
            }
          }
        }
        // should I comment out this "else"?
        //else {
        //  edm::LogWarning("L1CSCTPEmulatorWrongInput")
        //    << "+++ Warning in calcResolution(): no matched SimHit"
        //    << " found! +++\n";
        //}

        // "True bend", defined as difference in phi between muon hit
        // positions in the first and in the sixth layers.
        double phi1 = -999.0, phi6 = -999.0;
        vector<CSCCathodeLayerInfo>::const_iterator pli;
        for (pli = clctInfo.begin(); pli != clctInfo.end(); pli++) {
          CSCDetId layerId = pli->getId();
          int layer = layerId.layer();
          if (layer == 1 || layer == 6) {
            // Get simHits in this layer.
            for (const auto& psh : allSimHits) {
              // Find detId where simHit is located.
              CSCDetId hitId = (CSCDetId)psh.detUnitId();
              if (hitId == layerId &&
                  abs(psh.particleType()) == 13) { // muon hits only
                const CSCLayer* csclayer = geom_->layer(layerId);
                const auto& thisPoint = csclayer->toGlobal(psh.localPosition());
                double phi = thisPoint.phi();
                if (layer == 1)      phi1 = phi;
                else if (layer == 6) phi6 = phi;
                break; // simply take the first suitable hit.
              }
            }
          }
        }
        if (phi1 > -99. && phi6 > -99.) {
          double deltaPhi = phi1 - phi6;
          if (deltaPhi > M_PI)       deltaPhi -= 2.*M_PI;
          else if (deltaPhi < -M_PI) deltaPhi += 2.*M_PI;
          int csctype = getCSCType(id);
          hTrueBendCsc[csctype]->Fill(deltaPhi*1000.); // in mrad
        }
      }
    }
  }
}

void CSCTriggerPrimitivesReader::calcEfficiency(const CSCALCTDigiCollection* alcts,
                                                const CSCCLCTDigiCollection* clcts,
                                                const edm::PSimHitContainer& allSimHits) {

  // Book histos when called for the first time.
  if (!bookedEfficHistos) bookEfficHistos();

  // Create list of chambers having SimHits.
  vector<CSCDetId> chamberIds;
  vector<CSCDetId>::const_iterator chamberIdIt;
  for (const auto& simHitIt : allSimHits) {
    // Find detId where simHit is located.
    bool sameId = false;
    CSCDetId hitId = (CSCDetId) simHitIt.detUnitId();
    // Skip chambers marked as bad (includes most of ME4/2 chambers).
    if (checkBadChambers_ && badChambers_->isInBadChamber(hitId)) continue;
    //if (hitId.ring() == 4) continue; // skip ME1/A for now.
    if (!plotME1A && hitId.ring() == 4) continue;
    for (const auto& chamberId : chamberIds) {
      if (chamberId.endcap()  == hitId.endcap() &&
          chamberId.station() == hitId.station() &&
          chamberId.ring()    == hitId.ring() &&
          chamberId.chamber() == hitId.chamber()) {
        sameId = true;
        break;
      }
    }
    if (!sameId) {
      CSCDetId newChamberId(hitId.endcap(), hitId.station(), hitId.ring(),
                            hitId.chamber(), 0);
      chamberIds.push_back(newChamberId);
    }
  }
  LogTrace("CSCTriggerPrimitivesReader")
    << "Found SimHits in " << chamberIds.size() << " CSCs";

  bool used[CSCConstants::NUM_LAYERS];
  vector<PSimHit> simHitsV[CSCConstants::NUM_LAYERS];
  for (const auto& chamberId : chamberIds) {
    // Find out how many layers of this chamber have SimHits.
    int nLayers = 0;
    for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++) {
      used[ilayer] = false;
      simHitsV[ilayer].clear();
    }

    int endcap  = chamberId.endcap();
    int station = chamberId.station();
    int ring    = chamberId.ring();
    int chamber = chamberId.chamber();
    for (const auto& simHitIt : allSimHits) {
      CSCDetId hitId = (CSCDetId) simHitIt.detUnitId();
      if (hitId.endcap() == endcap && hitId.station() == station &&
          hitId.ring()   == ring   && hitId.chamber() == chamber) {
        int layer = hitId.layer() - 1;
        if (!used[layer] && abs(simHitIt.particleType()) == 13) {
          nLayers++;
          used[layer] = true;
          simHitsV[layer].push_back(simHitIt);
        }
      }
    }
    LogTrace("CSCTriggerPrimitivesReader")
      << "CSC in ME" << ((endcap == 1) ? "+" : "-")
      << station << "/" << ring << "/" << chamber
      << " has muon hits in " << nLayers << " layers";

    // If the number of layers with hits is above threshold, look for
    // a presence of LCTs.
    if (nLayers > 3) { // Should be a parameter.
      // Start with the key layer and take the eta of the first hit.
      // Really crude; should be improved.
      double hitEta = -999.;
      for (int ilayer = 2; ilayer < CSCConstants::NUM_LAYERS; ilayer++) {
        vector<PSimHit> layerSimHitsV = simHitsV[ilayer];
        if (layerSimHitsV.size() > 0) {
          LocalPoint hitLP = layerSimHitsV[0].localPosition();
          CSCDetId layerId = (CSCDetId)(layerSimHitsV[0]).detUnitId();
          const CSCLayer* csclayer = geom_->layer(layerId);
          GlobalPoint hitGP = csclayer->toGlobal(hitLP);
          hitEta = hitGP.eta();
          break;
        }
      }
      if (hitEta < -3.) {
        edm::LogWarning("L1CSCTPEmulatorWrongValues")
          << "+++ Warning in calcEfficiency(): no SimHit found"
          << " where there must be at least " << nLayers << "! +++\n";
        continue;
      }
      int csctype = getCSCType(*chamberIdIt);
      hEfficHitsEta[station-1]->Fill(fabs(hitEta));
      hEfficHitsEtaCsc[csctype]->Fill(fabs(hitEta));

      bool isALCT = false;
      for (auto adetUnitIt = alcts->begin(); adetUnitIt != alcts->end();
           adetUnitIt++) {
        const CSCDetId& id = (*adetUnitIt).first;
        if (id == (*chamberIdIt)) {
          const auto& range = (*adetUnitIt).second;
          for (auto digiIt = range.first;
               digiIt != range.second; digiIt++) {
            if (digiIt->isValid()) {
              // Check the distance??
              LogTrace("CSCTriggerPrimitivesReader") << "ALCT was found";
              isALCT = true;
              break;
            }
          }
        }
        if (isALCT) break;
      }
      if (isALCT) {
        hEfficALCTEta[station-1]->Fill(fabs(hitEta));
        hEfficALCTEtaCsc[csctype]->Fill(fabs(hitEta));
      }
      else {
        LogTrace("CSCTriggerPrimitivesReader") << "ALCT was not found";
      }

      bool isCLCT = false;
      for (auto cdetUnitIt = clcts->begin(); cdetUnitIt != clcts->end();
           cdetUnitIt++) {
        const CSCDetId& id = (*cdetUnitIt).first;
        if (id == (*chamberIdIt)) {
          const auto& range = (*cdetUnitIt).second;
          for (auto digiIt = range.first;
               digiIt != range.second; digiIt++) {
            if (digiIt->isValid()) {
              // Check the distance??
              LogTrace("CSCTriggerPrimitivesReader") << "CLCT was found";
              isCLCT = true;
              break;
            }
          }
        }
        if (isCLCT) break;
      }
      if (isCLCT) {
        hEfficCLCTEta[station-1]->Fill(fabs(hitEta));
        hEfficCLCTEtaCsc[csctype]->Fill(fabs(hitEta));
      }
      else {
        LogTrace("CSCTriggerPrimitivesReader") << "CLCT was not found";
      }

    }
  }
}

void CSCTriggerPrimitivesReader::drawALCTHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  string fname = resultsFileNamesPrefix_+"alcts.ps";
  TPostScript *ps = new TPostScript(fname.c_str(), 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[7], titl[50];
  TPaveLabel *title;

  int max_idh = plotME42 ? CSC_TYPES : CSC_TYPES-1;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of ALCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hAlctPerEvent->Draw();
  pad[page]->cd(2);  hAlctPerChamber->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hAlctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(3);  hAlctPerCSC->Draw();
  page++;  c1->Update();

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "ALCTs per chamber, endcap %d", endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    gStyle->SetOptStat(10);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hAlctCsc[endc][idh]->SetMinimum(0.0);
      pad[page]->cd(idh+1);  hAlctCsc[endc][idh]->Draw();
    }
    page++;  c1->Update();
  }

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hAlctValid->Draw();
  pad[page]->cd(2);  hAlctQuality->Draw();
  pad[page]->cd(3);  hAlctAccel->Draw();
  pad[page]->cd(4);  hAlctCollis->Draw();
  pad[page]->cd(5);  hAlctKeyGroup->Draw();
  pad[page]->cd(6);  hAlctBXN->Draw();
  page++;  c1->Update();

  ps->Close();
  delete c1;
}

void CSCTriggerPrimitivesReader::drawCLCTHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  string fname = resultsFileNamesPrefix_+"clcts.ps";
  TPostScript *ps = new TPostScript(fname.c_str(), 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[7], titl[50];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of CLCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hClctPerEvent->Draw();

  int max_idh = plotME42 ? CSC_TYPES : CSC_TYPES-1;

  edm::LogInfo("CSCTriggerPrimitivesReader") << "\n";
  int nbins = hClctPerChamber->GetNbinsX();
  for (int ibin = 1; ibin <= nbins; ibin++) {
    double f_bin = hClctPerChamber->GetBinContent(ibin);
    edm::LogInfo("CSCTriggerPrimitivesReader")
      << "  # CLCTs/chamber: " << ibin-1 << "; events: " << f_bin << endl;
  }

  pad[page]->cd(2);  hClctPerChamber->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hClctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(3);  hClctPerCSC->Draw();
  page++;  c1->Update();

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "CLCTs per chamber, endcap %d", endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    gStyle->SetOptStat(10);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hClctCsc[endc][idh]->SetMinimum(0.0);
      pad[page]->cd(idh+1);  hClctCsc[endc][idh]->Draw();
    }
    page++;  c1->Update();
  }

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hClctValid->Draw();
  pad[page]->cd(2);  hClctQuality->Draw();
  pad[page]->cd(3);  hClctSign->Draw();
  hClctPattern[1]->SetTitle("CLCT pattern #");
  pad[page]->cd(4);  hClctPattern[1]->Draw();
  pad[page]->cd(5);  hClctCFEB->Draw();
  pad[page]->cd(7);  hClctKeyStrip[1]->Draw();
  pad[page]->cd(8);  hClctBXN->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "CLCT bend for various chamber types, halfstrips");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    pad[page]->cd(idh+1);
    hClctBendCsc[idh][1]->GetXaxis()->SetTitle("Pattern bend");
    hClctBendCsc[idh][1]->GetYaxis()->SetTitle("Number of LCTs");
    hClctBendCsc[idh][1]->Draw();
  }
  page++;  c1->Update();

  ps->Close();
  delete c1;
}

void CSCTriggerPrimitivesReader::drawLCTTMBHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  string fname = resultsFileNamesPrefix_+"lcts_tmb.ps";
  TPostScript *ps = new TPostScript(fname.c_str(), 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[7], titl[50];
  TPaveLabel *title;

  int max_idh = plotME42 ? CSC_TYPES : CSC_TYPES-1;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of LCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  pad[page]->cd(1);  hLctTMBPerEvent->Draw();
  pad[page]->cd(2);  hLctTMBPerChamber->Draw();
  c1->Update();
  for (int i = 0; i < CSC_TYPES; i++) {
    hLctTMBPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    hCorrLctTMBPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(3);  hLctTMBPerCSC->Draw();
  pad[page]->cd(4);  hCorrLctTMBPerCSC->Draw();
  gStyle->SetOptStat(1110);
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT geometry");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctTMBEndcap->Draw();
  pad[page]->cd(2);  hLctTMBStation->Draw();
  pad[page]->cd(3);  hLctTMBSector->Draw();
  pad[page]->cd(4);  hLctTMBRing->Draw();
  for (int istat = 0; istat < MAX_STATIONS; istat++) {
    pad[page]->cd(istat+5);  hLctTMBChamber[istat]->Draw();
  }
  page++;  c1->Update();

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "LCTs per chamber, endcap %d", endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    gStyle->SetOptStat(10);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hLctTMBCsc[endc][idh]->SetMinimum(0.0);
      pad[page]->cd(idh+1);  hLctTMBCsc[endc][idh]->Draw();
    }
    page++;  c1->Update();
  }

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctTMBValid->Draw();
  pad[page]->cd(2);  hLctTMBQuality->Draw();
  pad[page]->cd(3);  hLctTMBKeyGroup->Draw();
  pad[page]->cd(4);  hLctTMBKeyStrip->Draw();
  pad[page]->cd(5);  hLctTMBStripType->Draw();
  pad[page]->cd(6);  hLctTMBPattern->Draw();
  pad[page]->cd(7);  hLctTMBBend->Draw();
  pad[page]->cd(8);  hLctTMBBXN->Draw();
  page++;  c1->Update();

  ps->Close();
  delete c1;
}

void CSCTriggerPrimitivesReader::drawLCTMPCHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  string fname = resultsFileNamesPrefix_+"lcts_mpc.ps";
  TPostScript *ps = new TPostScript(fname.c_str(), 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[7];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of LCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hLctMPCPerEvent->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hLctMPCPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    hCorrLctMPCPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hLctMPCPerCSC->Draw();
  pad[page]->cd(3);  hCorrLctMPCPerCSC->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT geometry");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctMPCEndcap->Draw();
  pad[page]->cd(2);  hLctMPCStation->Draw();
  pad[page]->cd(3);  hLctMPCSector->Draw();
  pad[page]->cd(4);  hLctMPCRing->Draw();
  for (int istat = 0; istat < MAX_STATIONS; istat++) {
    pad[page]->cd(istat+5);  hLctMPCChamber[istat]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctMPCValid->Draw();
  pad[page]->cd(2);  hLctMPCQuality->Draw();
  pad[page]->cd(3);  hLctMPCKeyGroup->Draw();
  pad[page]->cd(4);  hLctMPCKeyStrip->Draw();
  pad[page]->cd(5);  hLctMPCStripType->Draw();
  pad[page]->cd(6);  hLctMPCPattern->Draw();
  pad[page]->cd(7);  hLctMPCBend->Draw();
  pad[page]->cd(8);  hLctMPCBXN->Draw();
  page++;  c1->Update();

  ps->Close();
  delete c1;
}

void CSCTriggerPrimitivesReader::drawCompHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  string fname = resultsFileNamesPrefix_+"lcts_comp.ps";
  TPostScript *ps = new TPostScript(fname.c_str(), 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[7];
  TPaveLabel *title;
  Int_t nbins;

  TText teff;
  teff.SetTextFont(32);
  teff.SetTextSize(0.08);
  char eff[25], titl[60];

  int max_idh = plotME42 ? CSC_TYPES : CSC_TYPES-1;

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) { // endcaps
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "ALCT firmware-emulator: match in number found, endcap %d",
            endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    //gStyle->SetOptStat(110010);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hAlctFoundEffVsCsc[MAX_ENDCAPS][CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hAlctFoundEffVsCsc[endc][idh] =
        (TH1F*)hAlctCompFoundCsc[endc][idh]->Clone();
      hAlctFoundEffVsCsc[endc][idh]->Divide(hAlctCompSameNCsc[endc][idh],
                                            hAlctCompFoundCsc[endc][idh],
                                            1., 1., "B");
      nbins = hAlctCompFoundCsc[endc][idh]->GetNbinsX();
      for (Int_t ibin = 1; ibin <= nbins; ibin++) {
        if (hAlctCompFoundCsc[endc][idh]->GetBinContent(ibin) == 0) {
          hAlctFoundEffVsCsc[endc][idh]->SetBinContent(ibin, -1.);
          hAlctFoundEffVsCsc[endc][idh]->SetBinError(ibin, 0.);
        }
      }
      gPad->Update();  gStyle->SetStatX(0.65);
      hAlctFoundEffVsCsc[endc][idh]->SetMinimum(-0.05);
      hAlctFoundEffVsCsc[endc][idh]->SetMaximum(1.05);
      hAlctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitleOffset(0.7);
      hAlctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitleOffset(0.8);
      hAlctFoundEffVsCsc[endc][idh]->GetXaxis()->SetLabelSize(0.06); // default=0.04
      hAlctFoundEffVsCsc[endc][idh]->GetYaxis()->SetLabelSize(0.06); // default=0.04
      hAlctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitleSize(0.07); // default=0.05
      hAlctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitleSize(0.07); // default=0.05
      hAlctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitle("CSC id");
      hAlctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitle("% of same number found");
      pad[page]->cd(idh+1);  hAlctFoundEffVsCsc[endc][idh]->Draw("e");
      double numer = hAlctCompSameNCsc[endc][idh]->Integral();
      double denom = hAlctCompFoundCsc[endc][idh]->Integral();
      double ratio = 0.0, error = 0.0;
      if (denom > 0.) {
        ratio = numer/denom;
        error = sqrt(ratio*(1.-ratio)/denom);
      }
      sprintf(eff, "eff = (%4.1f +/- %4.1f)%%", ratio*100., error*100.);
      teff.DrawTextNDC(0.3, 0.5, eff);
    }
    page++;  c1->Update();
  }

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) { // endcaps
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "ALCT firmware-emulator: exact match, endcap %d", endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    //gStyle->SetOptStat(110010);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hAlctMatchEffVsCsc[MAX_ENDCAPS][CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hAlctMatchEffVsCsc[endc][idh] =
        (TH1F*)hAlctCompTotalCsc[endc][idh]->Clone();
      hAlctMatchEffVsCsc[endc][idh]->Divide(hAlctCompMatchCsc[endc][idh],
                                            hAlctCompTotalCsc[endc][idh],
                                            1., 1., "B");
      nbins = hAlctCompTotalCsc[endc][idh]->GetNbinsX();
      for (Int_t ibin = 1; ibin <= nbins; ibin++) {
        if (hAlctCompTotalCsc[endc][idh]->GetBinContent(ibin) == 0) {
          hAlctMatchEffVsCsc[endc][idh]->SetBinContent(ibin, -1.);
          hAlctMatchEffVsCsc[endc][idh]->SetBinError(ibin, 0.);
        }
      }
      gPad->Update();  gStyle->SetStatX(0.65);
      hAlctMatchEffVsCsc[endc][idh]->SetMinimum(-0.05);
      hAlctMatchEffVsCsc[endc][idh]->SetMaximum(1.05);
      hAlctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitleOffset(0.7);
      hAlctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitleOffset(0.8);
      hAlctMatchEffVsCsc[endc][idh]->GetXaxis()->SetLabelSize(0.06);
      hAlctMatchEffVsCsc[endc][idh]->GetYaxis()->SetLabelSize(0.06);
      hAlctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitleSize(0.07);
      hAlctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitleSize(0.07);
      hAlctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitle("CSC id");
      hAlctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitle("% of exact match");
      pad[page]->cd(idh+1);  hAlctMatchEffVsCsc[endc][idh]->Draw("e");
      double numer = hAlctCompMatchCsc[endc][idh]->Integral();
      double denom = hAlctCompTotalCsc[endc][idh]->Integral();
      double ratio = 0.0, error = 0.0;
      if (denom > 0.) {
        ratio = numer/denom;
        error = sqrt(ratio*(1.-ratio)/denom);
      }
      sprintf(eff, "eff = (%4.1f +/- %4.1f)%%", ratio*100., error*100.);
      teff.DrawTextNDC(0.3, 0.5, eff);
    }
    page++;  c1->Update();
  }

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "CLCT firmware-emulator: match in number found, endcap %d",
            endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    //gStyle->SetOptStat(110010);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hClctFoundEffVsCsc[MAX_ENDCAPS][CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hClctFoundEffVsCsc[endc][idh] =
        (TH1F*)hClctCompFoundCsc[endc][idh]->Clone();
      hClctFoundEffVsCsc[endc][idh]->Divide(hClctCompSameNCsc[endc][idh],
                                            hClctCompFoundCsc[endc][idh],
                                            1., 1., "B");
      nbins = hClctCompFoundCsc[endc][idh]->GetNbinsX();
      for (Int_t ibin = 1; ibin <= nbins; ibin++) {
        if (hClctCompFoundCsc[endc][idh]->GetBinContent(ibin) == 0) {
          hClctFoundEffVsCsc[endc][idh]->SetBinContent(ibin, -1.);
          hClctFoundEffVsCsc[endc][idh]->SetBinError(ibin, 0.);
        }
      }
      gPad->Update();  gStyle->SetStatX(0.65);
      hClctFoundEffVsCsc[endc][idh]->SetMinimum(-0.05);
      hClctFoundEffVsCsc[endc][idh]->SetMaximum(1.05);
      hClctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitleOffset(0.7);
      hClctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitleOffset(0.8);
      hClctFoundEffVsCsc[endc][idh]->GetXaxis()->SetLabelSize(0.06);
      hClctFoundEffVsCsc[endc][idh]->GetYaxis()->SetLabelSize(0.06);
      hClctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitleSize(0.07);
      hClctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitleSize(0.07);
      hClctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitle("CSC id");
      hClctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitle("% of same number found");
      pad[page]->cd(idh+1);  hClctFoundEffVsCsc[endc][idh]->Draw("e");
      double numer = hClctCompSameNCsc[endc][idh]->Integral();
      double denom = hClctCompFoundCsc[endc][idh]->Integral();
      double ratio = 0.0, error = 0.0;
      if (denom > 0.) {
        ratio = numer/denom;
        error = sqrt(ratio*(1.-ratio)/denom);
      }
      sprintf(eff, "eff = (%4.1f +/- %4.1f)%%", ratio*100., error*100.);
      teff.DrawTextNDC(0.3, 0.5, eff);
    }
    page++;  c1->Update();
  }

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "CLCT firmware-emulator: exact match, endcap %d", endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    //gStyle->SetOptStat(110010);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hClctMatchEffVsCsc[MAX_ENDCAPS][CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hClctMatchEffVsCsc[endc][idh] =
        (TH1F*)hClctCompTotalCsc[endc][idh]->Clone();
      hClctMatchEffVsCsc[endc][idh]->Divide(hClctCompMatchCsc[endc][idh],
                                            hClctCompTotalCsc[endc][idh],
                                            1., 1., "B");
      nbins = hClctCompTotalCsc[endc][idh]->GetNbinsX();
      for (Int_t ibin = 1; ibin <= nbins; ibin++) {
        if (hClctCompTotalCsc[endc][idh]->GetBinContent(ibin) == 0) {
          hClctMatchEffVsCsc[endc][idh]->SetBinContent(ibin, -1.);
          hClctMatchEffVsCsc[endc][idh]->SetBinError(ibin, 0.);
        }
      }
      gPad->Update();  gStyle->SetStatX(0.65);
      hClctMatchEffVsCsc[endc][idh]->SetMinimum(-0.05);
      hClctMatchEffVsCsc[endc][idh]->SetMaximum(1.05);
      hClctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitleOffset(0.7);
      hClctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitleOffset(0.8);
      hClctMatchEffVsCsc[endc][idh]->GetXaxis()->SetLabelSize(0.06);
      hClctMatchEffVsCsc[endc][idh]->GetYaxis()->SetLabelSize(0.06);
      hClctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitleSize(0.07);
      hClctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitleSize(0.07);
      hClctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitle("CSC id");
      hClctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitle("% of exact match");
      pad[page]->cd(idh+1);  hClctMatchEffVsCsc[endc][idh]->Draw("e");
      double numer = hClctCompMatchCsc[endc][idh]->Integral();
      double denom = hClctCompTotalCsc[endc][idh]->Integral();
      double ratio = 0.0, error = 0.0;
      if (denom > 0.) {
        ratio = numer/denom;
        error = sqrt(ratio*(1.-ratio)/denom);
      }
      sprintf(eff, "eff = (%4.1f +/- %4.1f)%%", ratio*100., error*100.);
      teff.DrawTextNDC(0.3, 0.5, eff);
    }
    page++;  c1->Update();
  }

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "LCT firmware-emulator: match in number found, endcap %d",
            endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    //gStyle->SetOptStat(110010);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hLctFoundEffVsCsc[MAX_ENDCAPS][CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hLctFoundEffVsCsc[endc][idh] =
        (TH1F*)hLctCompFoundCsc[endc][idh]->Clone();
      hLctFoundEffVsCsc[endc][idh]->Divide(hLctCompSameNCsc[endc][idh],
                                           hLctCompFoundCsc[endc][idh],
                                           1., 1., "B");
      nbins = hLctCompFoundCsc[endc][idh]->GetNbinsX();
      for (Int_t ibin = 1; ibin <= nbins; ibin++) {
        if (hLctCompFoundCsc[endc][idh]->GetBinContent(ibin) == 0) {
          hLctFoundEffVsCsc[endc][idh]->SetBinContent(ibin, -1.);
          hLctFoundEffVsCsc[endc][idh]->SetBinError(ibin, 0.);
        }
      }
      gPad->Update();  gStyle->SetStatX(0.65);
      hLctFoundEffVsCsc[endc][idh]->SetMinimum(-0.05);
      hLctFoundEffVsCsc[endc][idh]->SetMaximum(1.05);
      hLctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitleOffset(0.7);
      hLctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitleOffset(0.8);
      hLctFoundEffVsCsc[endc][idh]->GetXaxis()->SetLabelSize(0.06);
      hLctFoundEffVsCsc[endc][idh]->GetYaxis()->SetLabelSize(0.06);
      hLctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitleSize(0.07);
      hLctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitleSize(0.07);
      hLctFoundEffVsCsc[endc][idh]->GetXaxis()->SetTitle("CSC id");
      hLctFoundEffVsCsc[endc][idh]->GetYaxis()->SetTitle("% of same number found");
      pad[page]->cd(idh+1);  hLctFoundEffVsCsc[endc][idh]->Draw("e");
      double numer = hLctCompSameNCsc[endc][idh]->Integral();
      double denom = hLctCompFoundCsc[endc][idh]->Integral();
      double ratio = 0.0, error = 0.0;
      if (denom > 0.) {
        ratio = numer/denom;
        error = sqrt(ratio*(1.-ratio)/denom);
      }
      sprintf(eff, "eff = (%4.1f +/- %4.1f)%%", ratio*100., error*100.);
      teff.DrawTextNDC(0.3, 0.5, eff);
    }
    page++;  c1->Update();
  }

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    ps->NewPage();
    c1->Clear();  c1->cd(0);
    sprintf(titl, "LCT firmware-emulator: exact match, endcap %d", endc+1);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, titl);
    title->SetFillColor(10);  title->Draw();
    sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
    //gStyle->SetOptStat(110010);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hLctMatchEffVsCsc[MAX_ENDCAPS][CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hLctMatchEffVsCsc[endc][idh] =
        (TH1F*)hLctCompTotalCsc[endc][idh]->Clone();
      hLctMatchEffVsCsc[endc][idh]->Divide(hLctCompMatchCsc[endc][idh],
                                           hLctCompTotalCsc[endc][idh],
                                           1., 1., "B");
      nbins = hLctCompTotalCsc[endc][idh]->GetNbinsX();
      for (Int_t ibin = 1; ibin <= nbins; ibin++) {
        if (hLctCompTotalCsc[endc][idh]->GetBinContent(ibin) == 0) {
          hLctMatchEffVsCsc[endc][idh]->SetBinContent(ibin, -1.);
          hLctMatchEffVsCsc[endc][idh]->SetBinError(ibin, 0.);
        }
      }
      gPad->Update();  gStyle->SetStatX(0.65);
      hLctMatchEffVsCsc[endc][idh]->SetMinimum(-0.05);
      hLctMatchEffVsCsc[endc][idh]->SetMaximum(1.05);
      hLctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitleOffset(0.7);
      hLctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitleOffset(0.8);
      hLctMatchEffVsCsc[endc][idh]->GetXaxis()->SetLabelSize(0.06);
      hLctMatchEffVsCsc[endc][idh]->GetYaxis()->SetLabelSize(0.06);
      hLctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitleSize(0.07);
      hLctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitleSize(0.07);
      hLctMatchEffVsCsc[endc][idh]->GetXaxis()->SetTitle("CSC id");
      hLctMatchEffVsCsc[endc][idh]->GetYaxis()->SetTitle("% of exact match");
      pad[page]->cd(idh+1);  hLctMatchEffVsCsc[endc][idh]->Draw("e");
      double numer = hLctCompMatchCsc[endc][idh]->Integral();
      double denom = hLctCompTotalCsc[endc][idh]->Integral();
      double ratio = 0.0, error = 0.0;
      if (denom > 0.) {
        ratio = numer/denom;
        error = sqrt(ratio*(1.-ratio)/denom);
      }
      sprintf(eff, "eff = (%4.1f +/- %4.1f)%%", ratio*100., error*100.);
      teff.DrawTextNDC(0.3, 0.5, eff);
    }
    page++;  c1->Update();
  }

  ps->Close();
  delete c1;
}

void CSCTriggerPrimitivesReader::drawResolHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  string fname = resultsFileNamesPrefix_+"lcts_resol.ps";
  TPostScript *ps = new TPostScript(fname.c_str(), 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[7];
  TPaveLabel *title;

  int max_idh = plotME42 ? CSC_TYPES : CSC_TYPES-1;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT resolution");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.65);
  pad[page]->cd(1);  hEtaRecVsSim->SetMarkerSize(0.2);  hEtaRecVsSim->Draw();
  gPad->Update();  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.995);
  hResolDeltaWG->GetXaxis()->SetTitle("WG_{rec} - WG_{sim}");
  hResolDeltaWG->GetXaxis()->SetTitleOffset(1.2);
  hResolDeltaWG->GetYaxis()->SetTitle("Entries");
  hResolDeltaWG->GetYaxis()->SetTitleOffset(1.9);
  hResolDeltaWG->GetXaxis()->SetLabelSize(0.03);
  hResolDeltaWG->GetYaxis()->SetLabelSize(0.03);
  pad[page]->cd(3);  hResolDeltaWG->Draw();
  hResolDeltaEta->GetXaxis()->SetNdivisions(505); // twice fewer divisions
  hResolDeltaEta->GetXaxis()->SetLabelSize(0.04);
  hResolDeltaEta->GetYaxis()->SetLabelSize(0.04);
  pad[page]->cd(4);  hResolDeltaEta->Draw();  hResolDeltaEta->Fit("gaus","Q");
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#eta_rec-#eta_sim");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hEtaDiffCsc[idh][0]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hEtaDiffCsc[idh][0]->GetYaxis()->SetLabelSize(0.07);
    pad[page]->cd(idh+1);  hEtaDiffCsc[idh][0]->Draw();
    if (hEtaDiffCsc[idh][0]->GetEntries() > 1)
      hEtaDiffCsc[idh][0]->Fit("gaus","Q");
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#eta_rec-#eta_sim, endcap1");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hEtaDiffCsc[idh][1]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hEtaDiffCsc[idh][1]->GetYaxis()->SetLabelSize(0.07);
    pad[page]->cd(idh+1);  hEtaDiffCsc[idh][1]->Draw();
    if (hEtaDiffCsc[idh][1]->GetEntries() > 1)
      hEtaDiffCsc[idh][1]->Fit("gaus","Q");
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#eta_rec-#eta_sim, endcap2");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hEtaDiffCsc[idh][2]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hEtaDiffCsc[idh][2]->GetYaxis()->SetLabelSize(0.07);
    pad[page]->cd(idh+1);  hEtaDiffCsc[idh][2]->Draw();
    if (hEtaDiffCsc[idh][2]->GetEntries() > 1)
      hEtaDiffCsc[idh][2]->Fit("gaus","Q");
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#LT#eta_rec-#eta_sim#GT vs #eta_rec");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  //gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *hMeanEtaDiffVsEta[MAX_STATIONS];
  for (int istation = 0; istation < MAX_STATIONS; istation++) {
    hMeanEtaDiffVsEta[istation] = (TH1F*)hEtaDiffVsEta[istation]->Clone();
    hMeanEtaDiffVsEta[istation]->Divide(hEtaDiffVsEta[istation],
                                        hAlctVsEta[istation], 1., 1.);
    hMeanEtaDiffVsEta[istation]->GetXaxis()->SetTitleOffset(1.2);
    hMeanEtaDiffVsEta[istation]->GetXaxis()->SetTitle("#eta");
    hMeanEtaDiffVsEta[istation]->SetMaximum(0.05);
    pad[page]->cd(istation+1);  hMeanEtaDiffVsEta[istation]->Draw();
  }
  page++;  c1->Update();

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#eta_rec-#eta_sim vs wiregroup");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    pad[page]->cd(idh+1);  hEtaDiffVsWireCsc[idh]->SetMarkerSize(0.2);
    hEtaDiffVsWireCsc[idh]->GetXaxis()->SetTitle("Wiregroup");
    hEtaDiffVsWireCsc[idh]->GetXaxis()->SetTitleSize(0.07);
    hEtaDiffVsWireCsc[idh]->GetXaxis()->SetTitleOffset(1.2);
    hEtaDiffVsWireCsc[idh]->GetYaxis()->SetTitle("#eta_rec-#eta_sim");
    hEtaDiffVsWireCsc[idh]->GetYaxis()->SetTitleSize(0.07);
    hEtaDiffVsWireCsc[idh]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hEtaDiffVsWireCsc[idh]->GetYaxis()->SetLabelSize(0.07);
    hEtaDiffVsWireCsc[idh]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#phi resolution");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.65);
  pad[page]->cd(1);  hPhiRecVsSim->SetMarkerSize(0.2);  hPhiRecVsSim->Draw();
  gPad->Update();  gStyle->SetStatX(1.00);  gStyle->SetStatY(0.995);
  hResolDeltaPhi->GetXaxis()->SetLabelSize(0.04);
  hResolDeltaPhi->GetYaxis()->SetLabelSize(0.04);
  pad[page]->cd(2);  hResolDeltaPhi->Draw();  hResolDeltaPhi->Fit("gaus","Q");
  hResolDeltaHS->GetXaxis()->SetTitle("HS_{rec} - HS_{sim}");
  hResolDeltaHS->GetXaxis()->SetTitleOffset(1.2);
  hResolDeltaHS->GetYaxis()->SetTitle("Entries");
  hResolDeltaHS->GetYaxis()->SetTitleOffset(1.7);
  hResolDeltaHS->GetXaxis()->SetLabelSize(0.03);
  hResolDeltaHS->GetYaxis()->SetLabelSize(0.03);
  pad[page]->cd(3);  hResolDeltaHS->Draw();
  hResolDeltaDS->GetXaxis()->SetTitle("DS_{rec} - DS_{sim}");
  hResolDeltaDS->GetXaxis()->SetTitleOffset(1.2);
  hResolDeltaDS->GetYaxis()->SetTitle("Entries");
  hResolDeltaDS->GetYaxis()->SetTitleOffset(1.6);
  hResolDeltaDS->GetXaxis()->SetLabelSize(0.04);
  hResolDeltaDS->GetYaxis()->SetLabelSize(0.04);
  pad[page]->cd(4);  hResolDeltaDS->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "#phi_rec-#phi_sim (mrad)");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hPhiDiffCsc[idh][0]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hPhiDiffCsc[idh][0]->GetYaxis()->SetLabelSize(0.07);
    pad[page]->cd(idh+1);  hPhiDiffCsc[idh][0]->Draw();
    if (hPhiDiffCsc[idh][0]->GetEntries() > 1)
      hPhiDiffCsc[idh][0]->Fit("gaus","Q");
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#phi_rec-#phi_sim (mrad), endcap1");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hPhiDiffCsc[idh][1]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hPhiDiffCsc[idh][1]->GetYaxis()->SetLabelSize(0.07);
    pad[page]->cd(idh+1);  hPhiDiffCsc[idh][1]->Draw();
    if (hPhiDiffCsc[idh][1]->GetEntries() > 1)
      hPhiDiffCsc[idh][1]->Fit("gaus","Q");
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#phi_rec-#phi_sim (mrad), endcap2");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hPhiDiffCsc[idh][2]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hPhiDiffCsc[idh][2]->GetYaxis()->SetLabelSize(0.07);
    pad[page]->cd(idh+1);  hPhiDiffCsc[idh][2]->Draw();
    if (hPhiDiffCsc[idh][2]->GetEntries() > 1)
      hPhiDiffCsc[idh][2]->Fit("gaus","Q");
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#LT#phi_rec-#phi_sim#GT (mrad) vs #phi_rec");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *hMeanPhiDiffVsPhi[MAX_STATIONS];
  for (int istation = 0; istation < MAX_STATIONS; istation++) {
    hMeanPhiDiffVsPhi[istation] = (TH1F*)hPhiDiffVsPhi[istation]->Clone();
    hMeanPhiDiffVsPhi[istation]->Divide(hPhiDiffVsPhi[istation],
                                        hClctVsPhi[istation], 1., 1.);
    hMeanPhiDiffVsPhi[istation]->GetXaxis()->SetTitleOffset(1.2);
    hMeanPhiDiffVsPhi[istation]->GetYaxis()->SetTitleOffset(1.7);
    hMeanPhiDiffVsPhi[istation]->GetXaxis()->SetTitle("#phi");
    hMeanPhiDiffVsPhi[istation]->GetYaxis()->SetTitle("#LT#phi_rec-#phi_sim#GT (mrad)");
    hMeanPhiDiffVsPhi[istation]->SetMaximum(5.);
    pad[page]->cd(istation+1);  hMeanPhiDiffVsPhi[istation]->Draw();
  }
  page++;  c1->Update();

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#phi_rec-#phi_sim (mrad) vs halfstrip #");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    pad[page]->cd(idh+1);  hPhiDiffVsStripCsc[idh][1]->SetMarkerSize(0.2);
    hPhiDiffVsStripCsc[idh][1]->GetXaxis()->SetTitle("Halfstrip");
    hPhiDiffVsStripCsc[idh][1]->GetXaxis()->SetTitleOffset(1.4);
    hPhiDiffVsStripCsc[idh][1]->GetYaxis()->SetTitle("#phi_rec-#phi_sim (mrad)");
    hPhiDiffVsStripCsc[idh][1]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hPhiDiffVsStripCsc[idh][1]->GetYaxis()->SetLabelSize(0.07);
    hPhiDiffVsStripCsc[idh][1]->Draw();
  }
  page++;  c1->Update();

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#phi_rec-#phi_sim, halfstrips only, different patterns");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(3,3);
  int min_pattern, max_pattern;
  min_pattern = 2;
  max_pattern = CSCConstants::NUM_CLCT_PATTERNS;
  for (int idh = min_pattern; idh < max_pattern; idh++) {
    hPhiDiffPattern[idh]->GetXaxis()->SetTitle("Halfstrip");
    hPhiDiffPattern[idh]->GetXaxis()->SetTitleOffset(1.2);
    pad[page]->cd(idh-min_pattern+1);  hPhiDiffPattern[idh]->Draw();
    // if (hPhiDiffPattern[idh]->GetEntries() > 1)
    //   hPhiDiffPattern[idh]->Fit("gaus","Q");
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                         "#phi_1-#phi_6 (mrad), muon SimHits");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hTrueBendCsc[idh]->GetYaxis()->SetTitle("Entries");
    hTrueBendCsc[idh]->GetYaxis()->SetTitleSize(0.07);
    hTrueBendCsc[idh]->GetXaxis()->SetLabelSize(0.07); // default=0.04
    hTrueBendCsc[idh]->GetYaxis()->SetLabelSize(0.07);
    pad[page]->cd(idh+1);  hTrueBendCsc[idh]->Draw();
  }
  page++;  c1->Update();

  ps->Close();
  delete c1;
}

void CSCTriggerPrimitivesReader::drawEfficHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 700);
  string fname = resultsFileNamesPrefix_+"lcts_effic.ps";
  TPostScript *ps = new TPostScript(fname.c_str(), 111);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  int page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  char pagenum[7];
  TPaveLabel *title;
  char histtitle[60];

  gStyle->SetOptDate(0);
  gStyle->SetTitleSize(0.1, "");   // size for pad title; default is 0.02

  int max_idh = plotME42 ? CSC_TYPES : CSC_TYPES-1;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT efficiency vs #eta");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *hALCTEffVsEta[MAX_STATIONS];
  for (int istation = 0; istation < MAX_STATIONS; istation++) {
    hALCTEffVsEta[istation] = (TH1F*)hEfficHitsEta[istation]->Clone();
    hALCTEffVsEta[istation]->Divide(hEfficALCTEta[istation],
                                    hEfficHitsEta[istation], 1., 1., "B");
    hALCTEffVsEta[istation]->GetXaxis()->SetTitleOffset(1.2);
    hALCTEffVsEta[istation]->GetXaxis()->SetTitle("#eta");
    hALCTEffVsEta[istation]->SetMaximum(1.05);
    sprintf(histtitle, "ALCT efficiency vs #eta, station %d", istation+1);
    hALCTEffVsEta[istation]->SetTitle(histtitle);
    pad[page]->cd(istation+1);  hALCTEffVsEta[istation]->Draw();
  }
  page++;  c1->Update();

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT efficiency vs #eta");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(11111);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  TH1F *hALCTEffVsEtaCsc[CSC_TYPES];
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hALCTEffVsEtaCsc[idh] = (TH1F*)hEfficHitsEtaCsc[idh]->Clone();
    hALCTEffVsEtaCsc[idh]->Divide(hEfficALCTEtaCsc[idh],
                                  hEfficHitsEtaCsc[idh], 1., 1., "B");
    if (idh == 3 || idh == 4 || idh == 6 || idh == 8) {
      gPad->Update();  gStyle->SetStatX(0.43);
    }
    else {
      gPad->Update();  gStyle->SetStatX(1.00);
    }
    hALCTEffVsEtaCsc[idh]->GetXaxis()->SetTitle("#eta");
    hALCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleOffset(0.8);
    hALCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleSize(0.07); // default=0.05
    hALCTEffVsEtaCsc[idh]->GetXaxis()->SetLabelSize(0.10); // default=0.04
    hALCTEffVsEtaCsc[idh]->GetYaxis()->SetLabelSize(0.10);
    hALCTEffVsEtaCsc[idh]->SetLabelOffset(0.012, "XY");
    hALCTEffVsEtaCsc[idh]->SetMinimum(0.50);
    hALCTEffVsEtaCsc[idh]->SetMaximum(1.05);
    hALCTEffVsEtaCsc[idh]->SetTitle(csc_type[idh].c_str());
    hALCTEffVsEtaCsc[idh]->SetTitleSize(0.1, "");
    hALCTEffVsEtaCsc[idh]->SetLineWidth(2);
    hALCTEffVsEtaCsc[idh]->SetLineColor(4);
    pad[page]->cd(idh+1);  gPad->SetGrid(1);  hALCTEffVsEtaCsc[idh]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT efficiency vs #eta");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(0);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  TH1F *hCLCTEffVsEta[MAX_STATIONS];
  for (int istation = 0; istation < MAX_STATIONS; istation++) {
    hCLCTEffVsEta[istation] = (TH1F*)hEfficHitsEta[istation]->Clone();
    hCLCTEffVsEta[istation]->Divide(hEfficCLCTEta[istation],
                                    hEfficHitsEta[istation], 1., 1., "B");
    hCLCTEffVsEta[istation]->GetXaxis()->SetTitleOffset(1.2);
    hCLCTEffVsEta[istation]->GetXaxis()->SetTitle("#eta");
    hCLCTEffVsEta[istation]->SetMaximum(1.05);
    sprintf(histtitle, "CLCT efficiency vs #eta, station %d", istation+1);
    hCLCTEffVsEta[istation]->SetTitle(histtitle);
    pad[page]->cd(istation+1);  hCLCTEffVsEta[istation]->Draw();
  }
  page++;  c1->Update();

  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT efficiency vs #eta");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  TH1F *hCLCTEffVsEtaCsc[CSC_TYPES];
  for (int idh = 0; idh < max_idh; idh++) {
    if (!plotME1A && idh == 3) continue;
    hCLCTEffVsEtaCsc[idh] = (TH1F*)hEfficHitsEtaCsc[idh]->Clone();
    hCLCTEffVsEtaCsc[idh]->Divide(hEfficCLCTEtaCsc[idh],
                                  hEfficHitsEtaCsc[idh], 1., 1., "B");
    if (idh == 3 || idh == 4 || idh == 6 || idh == 8) {
      gPad->Update();  gStyle->SetStatX(0.43);
    }
    else {
      gPad->Update();  gStyle->SetStatX(1.00);
    }
    hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetTitle("#eta");
    hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleOffset(0.8);
    hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleSize(0.07); // default=0.05
    hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetLabelSize(0.10); // default=0.04
    hCLCTEffVsEtaCsc[idh]->GetYaxis()->SetLabelSize(0.10);
    hCLCTEffVsEtaCsc[idh]->SetLabelOffset(0.012, "XY");
    hCLCTEffVsEtaCsc[idh]->SetMinimum(0.50);
    hCLCTEffVsEtaCsc[idh]->SetMaximum(1.05);
    hCLCTEffVsEtaCsc[idh]->SetTitle(csc_type[idh].c_str());
    hCLCTEffVsEtaCsc[idh]->SetLineWidth(2);
    hCLCTEffVsEtaCsc[idh]->SetLineColor(4);
    pad[page]->cd(idh+1);  gPad->SetGrid(1);  hCLCTEffVsEtaCsc[idh]->Draw();
  }
  page++;  c1->Update();

  ps->Close();
  delete c1;
}

void CSCTriggerPrimitivesReader::drawHistosForTalks() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TCanvas *c2 = new TCanvas("c2", "", 0, 0, 540, 540);

  TPad *pad[MAXPAGES];
  for (int i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .07, .07, .93, .93);
  }

  int page = 1;
  TPaveLabel *title;
  gStyle->SetOptDate(0);

  int max_idh = plotME42 ? CSC_TYPES : CSC_TYPES-1;

  TPostScript *eps1 = new TPostScript("clcts.eps", 113);
  eps1->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT quantities");
  title->SetFillColor(10);  title->Draw();
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hClctQuality->Draw();
  pad[page]->cd(2);  hClctSign->Draw();
  TH1F* hClctPatternTot = (TH1F*)hClctPattern[0]->Clone();
  hClctPatternTot->SetTitle("CLCT pattern #");
  hClctPatternTot->Add(hClctPattern[0], hClctPattern[1], 1., 1.);
  pad[page]->cd(3);  hClctPatternTot->Draw();
  hClctPattern[0]->SetLineStyle(2);  hClctPattern[0]->Draw("same");
  hClctPattern[1]->SetLineStyle(3);  hClctPattern[1]->Draw("same");
  pad[page]->cd(4);  hClctCFEB->Draw();
  pad[page]->cd(5);  hClctStripType->Draw();
  TH1F* hClctKeyStripTot = (TH1F*)hClctKeyStrip[0]->Clone();
  hClctKeyStripTot->SetTitle("CLCT key strip #");
  hClctKeyStripTot->Add(hClctKeyStrip[0], hClctKeyStrip[1], 1., 1.);
  pad[page]->cd(6);  hClctKeyStripTot->Draw();
  page++;  c1->Update();
  c1->Print("asdf.png");
  eps1->Close();

  // Resolution histograms.
  if (bookedResolHistos) {
    gStyle->SetTitleSize(0.055, "");   // size for pad title; default is 0.02

    TPostScript *eps2 = new TPostScript("alct_deltaWG.eps", 113);
    eps2->NewPage();
    c2->Clear();  c2->cd(0);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(1,1);
    hResolDeltaWG->GetXaxis()->SetTitle("WG_{rec} - WG_{sim}");
    hResolDeltaWG->GetXaxis()->SetTitleOffset(1.2);
    hResolDeltaWG->GetYaxis()->SetTitle("Entries");
    hResolDeltaWG->GetYaxis()->SetTitleOffset(1.9);
    hResolDeltaWG->GetXaxis()->SetLabelSize(0.04);
    hResolDeltaWG->GetYaxis()->SetLabelSize(0.04);
    pad[page]->cd(1);  hResolDeltaWG->Draw();
    page++;  c2->Update();
    eps2->Close();

    TPostScript *eps3 = new TPostScript("clct_deltaHS.eps", 113);
    eps3->NewPage();
    c2->Clear();  c2->cd(0);
    pad[page]->Draw();
    pad[page]->Divide(1,1);
    hResolDeltaHS->GetXaxis()->SetTitle("HS_{rec} - HS_{sim}");
    hResolDeltaHS->GetXaxis()->SetTitleOffset(1.2);
    hResolDeltaHS->GetYaxis()->SetTitle("Entries");
    hResolDeltaHS->GetYaxis()->SetTitleOffset(1.7);
    hResolDeltaHS->GetXaxis()->SetLabelSize(0.04); // default=0.04
    hResolDeltaHS->GetYaxis()->SetLabelSize(0.04);
    pad[page]->cd(1);  hResolDeltaHS->Draw();
    page++;  c2->Update();
    eps3->Close();

    TPostScript *eps4 = new TPostScript("clct_deltaDS.eps", 113);
    eps4->NewPage();
    c2->Clear();  c2->cd(0);
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(1,1);
    hResolDeltaDS->GetXaxis()->SetTitle("DS_{rec} - DS_{sim}");
    hResolDeltaDS->GetXaxis()->SetTitleOffset(1.2);
    hResolDeltaDS->GetYaxis()->SetTitle("Entries");
    hResolDeltaDS->GetYaxis()->SetTitleOffset(1.6);
    hResolDeltaDS->GetXaxis()->SetLabelSize(0.04); // default=0.04
    hResolDeltaDS->GetYaxis()->SetLabelSize(0.04);
    pad[page]->cd(1);  hResolDeltaDS->Draw();
    page++;  c2->Update();
    eps4->Close();

    TPostScript *eps5 = new TPostScript("clct_deltaPhi_hs.eps", 113);
    eps5->NewPage();
    c1->Clear();  c1->cd(0);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
                           "#phi_rec-#phi_sim (mrad), halfstrips only");
    title->SetFillColor(10);  title->Draw();
    gStyle->SetTitleSize(0.1, "");   // size for pad title; default is 0.02
    gStyle->SetOptStat(111110);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      pad[page]->cd(idh+1);  hPhiDiffCsc[idh][4]->Draw();
      //if (hPhiDiffCsc[idh][4]->GetEntries() > 1)
      //hPhiDiffCsc[idh][4]->Fit("gaus","Q");
      //hPhiDiffCsc[idh][4]->GetXaxis()->SetTitle("#phi_{rec} - #phi_{sim} (mrad)");
      //hPhiDiffCsc[idh][4]->GetXaxis()->SetTitleSize(0.06);
      //hPhiDiffCsc[idh][4]->GetXaxis()->SetTitleOffset(0.9);
      hPhiDiffCsc[idh][4]->GetYaxis()->SetTitle("Entries        ");
      hPhiDiffCsc[idh][4]->GetYaxis()->SetTitleSize(0.07);
      hPhiDiffCsc[idh][4]->GetYaxis()->SetTitleOffset(1.0);
      hPhiDiffCsc[idh][4]->GetXaxis()->SetLabelSize(0.10); // default=0.04
      hPhiDiffCsc[idh][4]->GetYaxis()->SetLabelSize(0.10);
      hPhiDiffCsc[idh][4]->SetLabelOffset(0.012, "XY");
    }
    page++;  c1->Update();
    eps5->Close();
  }

  // Efficiency histograms.
  if (bookedEfficHistos) {
    TPostScript *eps6 = new TPostScript("alct_effic.eps", 113);
    eps6->NewPage();
    c1->Clear();  c1->cd(0);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT efficiency vs #eta");
    title->SetFillColor(10);  title->Draw();
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hALCTEffVsEtaCsc[CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hALCTEffVsEtaCsc[idh] = (TH1F*)hEfficHitsEtaCsc[idh]->Clone();
      hALCTEffVsEtaCsc[idh]->Divide(hEfficALCTEtaCsc[idh],
                                    hEfficHitsEtaCsc[idh], 1., 1., "B");
      hALCTEffVsEtaCsc[idh]->GetXaxis()->SetTitle("#eta");
      hALCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleOffset(0.8);
      hALCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleSize(0.07); // default=0.05
      hALCTEffVsEtaCsc[idh]->GetXaxis()->SetLabelSize(0.10); // default=0.04
      hALCTEffVsEtaCsc[idh]->GetYaxis()->SetLabelSize(0.10);
      hALCTEffVsEtaCsc[idh]->SetLabelOffset(0.012, "XY");
      hALCTEffVsEtaCsc[idh]->SetMinimum(0.50);
      hALCTEffVsEtaCsc[idh]->SetMaximum(1.05);
      hALCTEffVsEtaCsc[idh]->SetTitle(csc_type[idh].c_str());
      hALCTEffVsEtaCsc[idh]->SetTitleSize(0.1, "");
      hALCTEffVsEtaCsc[idh]->SetLineWidth(2);
      hALCTEffVsEtaCsc[idh]->SetLineColor(4);
      hALCTEffVsEtaCsc[idh]->SetLineColor(4);
      pad[page]->cd(idh+1);  gPad->SetGrid(1);  hALCTEffVsEtaCsc[idh]->Draw();
    }
    page++;  c1->Update();
    eps6->Close();

    TPostScript *eps7 = new TPostScript("clct_effic.eps", 113);
    eps7->NewPage();
    c1->Clear();  c1->cd(0);
    title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT efficiency vs #eta");
    title->SetFillColor(10);  title->Draw();
    gStyle->SetOptStat(0);
    pad[page]->Draw();
    pad[page]->Divide(2,5);
    TH1F *hCLCTEffVsEtaCsc[CSC_TYPES];
    for (int idh = 0; idh < max_idh; idh++) {
      if (!plotME1A && idh == 3) continue;
      hCLCTEffVsEtaCsc[idh] = (TH1F*)hEfficHitsEtaCsc[idh]->Clone();
      hCLCTEffVsEtaCsc[idh]->Divide(hEfficCLCTEtaCsc[idh],
                                    hEfficHitsEtaCsc[idh], 1., 1., "B");
      hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetTitle("#eta");
      hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleOffset(0.8);
      hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetTitleSize(0.07); // default=0.05
      hCLCTEffVsEtaCsc[idh]->GetXaxis()->SetLabelSize(0.10); // default=0.04
      hCLCTEffVsEtaCsc[idh]->GetYaxis()->SetLabelSize(0.10);
      hCLCTEffVsEtaCsc[idh]->SetLabelOffset(0.012, "XY");
      hCLCTEffVsEtaCsc[idh]->SetMinimum(0.50);
      hCLCTEffVsEtaCsc[idh]->SetMaximum(1.05);
      hCLCTEffVsEtaCsc[idh]->SetTitle(csc_type[idh].c_str());
      hCLCTEffVsEtaCsc[idh]->SetLineWidth(2);
      hCLCTEffVsEtaCsc[idh]->SetLineColor(4);
      pad[page]->cd(idh+1);  gPad->SetGrid(1);  hCLCTEffVsEtaCsc[idh]->Draw();
    }
    page++;  c1->Update();
    eps7->Close();
  }
  delete c1;
  delete c2;
}

GlobalPoint
CSCTriggerPrimitivesReader::getGlobalPosition(unsigned int rawId, int keyWg, int keyHS) const
{
	// taken from https://github.com/cms-sw/cmssw/blob/dc9f78b6af4ad56c9342cf14041b6485a60b0691/L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesReaderGEM.cc
	CSCDetId cscId = CSCDetId(rawId);
	int ring = cscId.ring();
	if (cscId.station() == 1 and cscId.ring() == 1 and (lut_wg_vs_hs_me1b[keyWg][0] <0 || keyHS>=128)){
	  ring =4;
	  if (keyHS >= 128)
	    keyHS = keyHS-128;
	}

	CSCDetId key_id(cscId.endcap(), cscId.station(), ring,
                  cscId.chamber(), CSCConstants::KEY_CLCT_LAYER);
	auto cscChamber = geom_->chamber(cscId);
	float fractional_strip = 0.5 * (keyHS + 1) - 0.25;
	auto layer_geo = cscChamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
	// LCT::getKeyWG() also starts from 0
	float wire = layer_geo->middleWireOfGroup(keyWg + 1);
	LocalPoint csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
	GlobalPoint csc_gp = geom_->idToDet(key_id)->surface().toGlobal(csc_intersect);
	return csc_gp;

}


bool CSCTriggerPrimitivesReader::doesALCTCrossCLCT(CSCDetId id, int key_wg, int key_hs) const
{

  bool isME11 = (id.station() ==1 and (id.ring()==1 or id.ring()==4));
  if (not isME11) return true;

  int theEndcap = (id.endcap() == 1)?1 : 2;

  if (key_hs > CSCConstants::MAX_HALF_STRIP_ME1B)
    {
      key_hs = key_hs - CSCConstants::MAX_HALF_STRIP_ME1B -1;//convert it from 128-223 -> 0-95
      if ( !gangedME1a )
        {
          // wrap around ME11 HS number for -z endcap
          if (theEndcap==2) key_hs = CSCConstants::MAX_HALF_STRIP_ME1A_UNGANGED - key_hs;
          if ( key_hs >= lut_wg_vs_hs_me1a[key_wg][0] &&
               key_hs <= lut_wg_vs_hs_me1a[key_wg][1]    ) return true;
          return false;
        }
      else
        {
          if (theEndcap==2) key_hs = CSCConstants::MAX_HALF_STRIP_ME1A_GANGED - key_hs;
          if ( key_hs >= lut_wg_vs_hs_me1ag[key_wg][0] &&
               key_hs <= lut_wg_vs_hs_me1ag[key_wg][1]    ) return true;
          return false;
        }
    }
  if ( key_hs <= CSCConstants::MAX_HALF_STRIP_ME1B)
    {
      if (theEndcap==2) key_hs = CSCConstants::MAX_HALF_STRIP_ME1B - key_hs;
      if ( key_hs >= lut_wg_vs_hs_me1b[key_wg][0] &&
           key_hs <= lut_wg_vs_hs_me1b[key_wg][1]      ) return true;
    }
  return false;
}




// Returns chamber type (0-9) according to the station and ring number
int CSCTriggerPrimitivesReader::getCSCType(const CSCDetId& id) {
  int type = -999;

  if (id.station() == 1) {
    type = (id.triggerCscId()-1)/3;
    if (id.ring() == 4) {
      type = 3;
    }
  }
  else { // stations 2-4
    type = 3 + id.ring() + 2*(id.station()-2);
  }

  assert(type >= 0 && type < CSC_TYPES);
  return type;
}

// Returns halfstrips-per-radian for different CSC types
double CSCTriggerPrimitivesReader::getHsPerRad(const int idh) {
  return (NCHAMBERS[idh]*MAX_HS[idh]/TWOPI);
}

DEFINE_FWK_MODULE(CSCTriggerPrimitivesReader);
//-------------------------------------------------
