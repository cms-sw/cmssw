// -*- C++ -*-
//
// Package:    GEMSimHitAnalyzer
// Class:      GEMSimHitAnalyzer
// 
// \class GEMSimHitAnalyzer
//
// Description: Analyzer GEM SimHit information (as well as CSC & RPC SimHits). 
// To be used for GEM algorithm development.
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"

#include "RPCGEM/GEMValidation/interface/GEMSimTracksProcessor.h"

using namespace std;


struct MyCSCSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t endcap, ring, station, chamber, layer;
  Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
  Float_t Phi_0, DeltaPhi, R_0;
};

struct MyRPCSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t region, ring, station, sector, layer, subsector, roll;
  Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
  Int_t strip;
};


struct MyGEMSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t region, ring, station, layer, chamber, roll;
  Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
  Int_t strip;
  Float_t Phi_0, DeltaPhi, R_0;
};

struct MySimTrack
{
  Float_t meanSimHitRhoGEMl1Even, meanSimHitEtaGEMl1Even, meanSimHitPhiGEMl1Even;
  Float_t meanSimHitRhoGEMl2Even, meanSimHitEtaGEMl2Even, meanSimHitPhiGEMl2Even;
  Float_t meanSimHitRhoCSCEven, meanSimHitEtaCSCEven, meanSimHitPhiCSCEven;
  Float_t meanSimHitRhoGEMl1Odd, meanSimHitEtaGEMl1Odd, meanSimHitPhiGEMl1Odd;
  Float_t meanSimHitRhoGEMl2Odd, meanSimHitEtaGEMl2Odd, meanSimHitPhiGEMl2Odd;
  Float_t meanSimHitRhoCSCOdd, meanSimHitEtaCSCOdd, meanSimHitPhiCSCOdd;
  Float_t meanSimHitRhoGEMl1Both, meanSimHitEtaGEMl1Both, meanSimHitPhiGEMl1Both;
  Float_t meanSimHitRhoGEMl2Both, meanSimHitEtaGEMl2Both, meanSimHitPhiGEMl2Both;
  Float_t meanSimHitRhoCSCBoth, meanSimHitEtaCSCBoth, meanSimHitPhiCSCBoth;
  Float_t propagatedSimHitRhoGEMl1Even, propagatedSimHitEtaGEMl1Even, propagatedSimHitPhiGEMl1Even;
  Float_t propagatedSimHitRhoGEMl2Even, propagatedSimHitEtaGEMl2Even, propagatedSimHitPhiGEMl2Even;
  Float_t propagatedSimHitRhoCSCEven, propagatedSimHitEtaCSCEven, propagatedSimHitPhiCSCEven;
  Float_t propagatedSimHitRhoGEMl1Odd, propagatedSimHitEtaGEMl1Odd, propagatedSimHitPhiGEMl1Odd;
  Float_t propagatedSimHitRhoGEMl2Odd, propagatedSimHitEtaGEMl2Odd, propagatedSimHitPhiGEMl2Odd;
  Float_t propagatedSimHitRhoCSCOdd, propagatedSimHitEtaCSCOdd, propagatedSimHitPhiCSCOdd;
  Float_t propagatedSimHitRhoGEMl1Both, propagatedSimHitEtaGEMl1Both, propagatedSimHitPhiGEMl1Both;
  Float_t propagatedSimHitRhoGEMl2Both, propagatedSimHitEtaGEMl2Both, propagatedSimHitPhiGEMl2Both;
  Float_t propagatedSimHitRhoCSCBoth, propagatedSimHitEtaCSCBoth, propagatedSimHitPhiCSCBoth;
  Float_t charge, pt, eta, phi;
  Int_t hasGEMl1, hasGEMl2, hasCSC;
};


class GEMSimHitAnalyzer : public edm::EDAnalyzer
{
public:
  /// Constructor
  explicit GEMSimHitAnalyzer(const edm::ParameterSet& iConfig);
  /// Destructor
  ~GEMSimHitAnalyzer();
  
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  
  void bookCSCSimHitsTree();
  void bookRPCSimHitsTree();
  void bookGEMSimHitsTree();
  void bookSimTracksTree();
    
  void analyzeCSC( const edm::Event& iEvent );
  void analyzeRPC( const edm::Event& iEvent );
  void analyzeGEM( const edm::Event& iEvent );
  void analyzeTracks();
  
  std::string simInputLabel;

  GEMSimTracksProcessor* simTrkProcessor;

  TTree* csc_sh_tree;
  TTree* rpc_sh_tree;
  TTree* gem_sh_tree;
  TTree* track_tree;
  
  edm::Handle<edm::PSimHitContainer> CSCHits;
  edm::Handle<edm::PSimHitContainer> RPCHits;
  edm::Handle<edm::PSimHitContainer> GEMHits;
  edm::Handle<edm::SimTrackContainer> simTracks;
  edm::Handle<edm::SimVertexContainer> simVertices;
  
  edm::ESHandle<CSCGeometry> csc_geom;
  edm::ESHandle<RPCGeometry> rpc_geom;
  edm::ESHandle<GEMGeometry> gem_geom;
  
  const CSCGeometry* csc_geometry;
  const RPCGeometry* rpc_geometry;
  const GEMGeometry* gem_geometry;
  
  MyCSCSimHit csc_sh;
  MyRPCSimHit rpc_sh;
  MyGEMSimHit gem_sh;
  MySimTrack  track;
  GlobalPoint hitGP;
 
};

// Constructor
GEMSimHitAnalyzer::GEMSimHitAnalyzer(const edm::ParameterSet& iConfig)
{
  simInputLabel = iConfig.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits");
  simTrkProcessor = new GEMSimTracksProcessor(iConfig);
  
  bookCSCSimHitsTree();
  bookRPCSimHitsTree();
  bookGEMSimHitsTree();  
  bookSimTracksTree();
}


GEMSimHitAnalyzer::~GEMSimHitAnalyzer() {}


void GEMSimHitAnalyzer::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
{
  simTrkProcessor->init(iSetup);
}


void GEMSimHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(csc_geom);
  csc_geometry = &*csc_geom;
  
  iSetup.get<MuonGeometryRecord>().get(rpc_geom);
  rpc_geometry = &*rpc_geom;
  
  iSetup.get<MuonGeometryRecord>().get(gem_geom);
  gem_geometry = &*gem_geom;
  
  iEvent.getByLabel(simInputLabel, simTracks);
  iEvent.getByLabel(simInputLabel, simVertices);

  simTrkProcessor->fillTracks( *(simTracks.product()), *(simVertices.product()) );

  iEvent.getByLabel(edm::InputTag(simInputLabel,"MuonCSCHits"), CSCHits);
  if(CSCHits->size()) analyzeCSC( iEvent );
  
  iEvent.getByLabel(edm::InputTag(simInputLabel,"MuonRPCHits"), RPCHits);
  if(RPCHits->size()) analyzeRPC( iEvent );
  
  iEvent.getByLabel(edm::InputTag(simInputLabel,"MuonGEMHits"), GEMHits);
  if(GEMHits->size()) analyzeGEM( iEvent );  
 

  if(simTrkProcessor->size()) analyzeTracks();
}


void GEMSimHitAnalyzer::bookCSCSimHitsTree()
{  
  edm::Service<TFileService> fs;
  csc_sh_tree = fs->make<TTree>("CSCSimHits", "CSCSimHits");
  csc_sh_tree->Branch("eventNumber",&csc_sh.eventNumber);
  csc_sh_tree->Branch("detUnitId",&csc_sh.detUnitId);
  csc_sh_tree->Branch("particleType",&csc_sh.particleType);
  csc_sh_tree->Branch("x",&csc_sh.x);
  csc_sh_tree->Branch("y",&csc_sh.y);
  csc_sh_tree->Branch("energyLoss",&csc_sh.energyLoss);
  csc_sh_tree->Branch("pabs",&csc_sh.pabs);
  csc_sh_tree->Branch("timeOfFlight",&csc_sh.timeOfFlight);
  csc_sh_tree->Branch("timeOfFlight",&csc_sh.timeOfFlight);
  csc_sh_tree->Branch("endcap",&csc_sh.endcap);
  csc_sh_tree->Branch("ring",&csc_sh.ring);
  csc_sh_tree->Branch("station",&csc_sh.station);
  csc_sh_tree->Branch("chamber",&csc_sh.chamber);
  csc_sh_tree->Branch("layer",&csc_sh.layer);
  csc_sh_tree->Branch("globalR",&csc_sh.globalR);
  csc_sh_tree->Branch("globalEta",&csc_sh.globalEta);
  csc_sh_tree->Branch("globalPhi",&csc_sh.globalPhi);
  csc_sh_tree->Branch("globalX",&csc_sh.globalX);
  csc_sh_tree->Branch("globalY",&csc_sh.globalY);
  csc_sh_tree->Branch("globalZ",&csc_sh.globalZ);
  csc_sh_tree->Branch("Phi_0", &csc_sh.Phi_0);
  csc_sh_tree->Branch("DeltaPhi", &csc_sh.DeltaPhi);
  csc_sh_tree->Branch("R_0", &csc_sh.R_0);
}


void GEMSimHitAnalyzer::bookRPCSimHitsTree()
{
  edm::Service< TFileService > fs;
  rpc_sh_tree = fs->make< TTree >("RPCSimHits", "RPCSimHits");
  rpc_sh_tree->Branch("eventNumber", &rpc_sh.eventNumber);
  rpc_sh_tree->Branch("detUnitId", &rpc_sh.detUnitId);
  rpc_sh_tree->Branch("particleType", &rpc_sh.particleType);
  rpc_sh_tree->Branch("x", &rpc_sh.x);
  rpc_sh_tree->Branch("y", &rpc_sh.y);
  rpc_sh_tree->Branch("energyLoss", &rpc_sh.energyLoss);
  rpc_sh_tree->Branch("pabs", &rpc_sh.pabs);
  rpc_sh_tree->Branch("timeOfFlight", &rpc_sh.timeOfFlight);
  rpc_sh_tree->Branch("timeOfFlight", &rpc_sh.timeOfFlight);
  rpc_sh_tree->Branch("ring", &rpc_sh.ring);
  rpc_sh_tree->Branch("station", &rpc_sh.station);
  rpc_sh_tree->Branch("layer", &rpc_sh.layer);
  rpc_sh_tree->Branch("globalR", &rpc_sh.globalR);
  rpc_sh_tree->Branch("globalEta", &rpc_sh.globalEta);
  rpc_sh_tree->Branch("globalPhi", &rpc_sh.globalPhi);
  rpc_sh_tree->Branch("globalX", &rpc_sh.globalX);
  rpc_sh_tree->Branch("globalY", &rpc_sh.globalY);
  rpc_sh_tree->Branch("globalZ", &rpc_sh.globalZ);
  rpc_sh_tree->Branch("strip", &rpc_sh.strip);
}


void GEMSimHitAnalyzer::bookGEMSimHitsTree()
{
  edm::Service< TFileService > fs;
  gem_sh_tree = fs->make< TTree >("GEMSimHits", "GEMSimHits");
  gem_sh_tree->Branch("eventNumber", &gem_sh.eventNumber);
  gem_sh_tree->Branch("detUnitId", &gem_sh.detUnitId);
  gem_sh_tree->Branch("particleType", &gem_sh.particleType);
  gem_sh_tree->Branch("x", &gem_sh.x);
  gem_sh_tree->Branch("y", &gem_sh.y);
  gem_sh_tree->Branch("energyLoss", &gem_sh.energyLoss);
  gem_sh_tree->Branch("pabs", &gem_sh.pabs);
  gem_sh_tree->Branch("timeOfFlight", &gem_sh.timeOfFlight);
  gem_sh_tree->Branch("region", &gem_sh.region);
  gem_sh_tree->Branch("ring", &gem_sh.ring);
  gem_sh_tree->Branch("station", &gem_sh.station);
  gem_sh_tree->Branch("chamber", &gem_sh.chamber);
  gem_sh_tree->Branch("layer", &gem_sh.layer);
  gem_sh_tree->Branch("roll", &gem_sh.roll);
  gem_sh_tree->Branch("globalR", &gem_sh.globalR);
  gem_sh_tree->Branch("globalEta", &gem_sh.globalEta);
  gem_sh_tree->Branch("globalPhi", &gem_sh.globalPhi);
  gem_sh_tree->Branch("globalX", &gem_sh.globalX);
  gem_sh_tree->Branch("globalY", &gem_sh.globalY);
  gem_sh_tree->Branch("globalZ", &gem_sh.globalZ);
  gem_sh_tree->Branch("strip", &gem_sh.strip);
  gem_sh_tree->Branch("Phi_0", &gem_sh.Phi_0);
  gem_sh_tree->Branch("DeltaPhi", &gem_sh.DeltaPhi);
  gem_sh_tree->Branch("R_0", &gem_sh.R_0);
}


void GEMSimHitAnalyzer::bookSimTracksTree()
{
  edm::Service< TFileService > fs;
  track_tree = fs->make< TTree >("Tracks", "Tracks");
  track_tree->Branch("meanSimHitRhoGEMl1Even",&track.meanSimHitRhoGEMl1Even);
  track_tree->Branch("meanSimHitEtaGEMl1Even",&track.meanSimHitEtaGEMl1Even);
  track_tree->Branch("meanSimHitPhiGEMl1Even",&track.meanSimHitPhiGEMl1Even);
  track_tree->Branch("meanSimHitRhoGEMl2Even",&track. meanSimHitRhoGEMl2Even);
  track_tree->Branch("meanSimHitEtaGEMl2Even",&track.meanSimHitEtaGEMl2Even);
  track_tree->Branch("meanSimHitPhiGEMl2Even",&track.meanSimHitPhiGEMl2Even);
  track_tree->Branch("meanSimHitRhoCSCEven",&track. meanSimHitRhoCSCEven);
  track_tree->Branch("meanSimHitEtaCSCEven",&track.meanSimHitEtaCSCEven);
  track_tree->Branch("meanSimHitPhiCSCEven",&track.meanSimHitPhiCSCEven);
  track_tree->Branch("propagatedSimHitRhoGEMl1Even",&track. propagatedSimHitRhoGEMl1Even);
  track_tree->Branch("propagatedSimHitEtaGEMl1Even",&track.propagatedSimHitEtaGEMl1Even);
  track_tree->Branch("propagatedSimHitPhiGEMl1Even",&track.propagatedSimHitPhiGEMl1Even);
  track_tree->Branch("propagatedSimHitRhoGEMl2Even",&track. propagatedSimHitRhoGEMl2Even);
  track_tree->Branch("propagatedSimHitEtaGEMl2Even",&track.propagatedSimHitEtaGEMl2Even);
  track_tree->Branch("propagatedSimHitPhiGEMl2Even",&track.propagatedSimHitPhiGEMl2Even);
  track_tree->Branch("propagatedSimHitRhoCSCEven",&track. propagatedSimHitRhoCSCEven);
  track_tree->Branch("propagatedSimHitEtaCSCEven",&track.propagatedSimHitEtaCSCEven);
  track_tree->Branch("propagatedSimHitPhiCSCEven",&track.propagatedSimHitPhiCSCEven);

  track_tree->Branch("meanSimHitRhoGEMl1Odd",&track. meanSimHitRhoGEMl1Odd);
  track_tree->Branch("meanSimHitEtaGEMl1Odd",&track.meanSimHitEtaGEMl1Odd);
  track_tree->Branch("meanSimHitPhiGEMl1Odd",&track.meanSimHitPhiGEMl1Odd);
  track_tree->Branch("meanSimHitRhoGEMl2Odd",&track. meanSimHitRhoGEMl2Odd);
  track_tree->Branch("meanSimHitEtaGEMl2Odd",&track.meanSimHitEtaGEMl2Odd);
  track_tree->Branch("meanSimHitPhiGEMl2Odd",&track.meanSimHitPhiGEMl2Odd);
  track_tree->Branch("meanSimHitRhoCSCOdd",&track. meanSimHitRhoCSCOdd);
  track_tree->Branch("meanSimHitEtaCSCOdd",&track.meanSimHitEtaCSCOdd);
  track_tree->Branch("meanSimHitPhiCSCOdd",&track.meanSimHitPhiCSCOdd);
  track_tree->Branch("propagatedSimHitRhoGEMl1Odd",&track. propagatedSimHitRhoGEMl1Odd);
  track_tree->Branch("propagatedSimHitEtaGEMl1Odd",&track.propagatedSimHitEtaGEMl1Odd);
  track_tree->Branch("propagatedSimHitPhiGEMl1Odd",&track.propagatedSimHitPhiGEMl1Odd);
  track_tree->Branch("propagatedSimHitRhoGEMl2Odd",&track. propagatedSimHitRhoGEMl2Odd);
  track_tree->Branch("propagatedSimHitEtaGEMl2Odd",&track.propagatedSimHitEtaGEMl2Odd);
  track_tree->Branch("propagatedSimHitPhiGEMl2Odd",&track.propagatedSimHitPhiGEMl2Odd);
  track_tree->Branch("propagatedSimHitRhoCSCOdd",&track. propagatedSimHitRhoCSCOdd);
  track_tree->Branch("propagatedSimHitEtaCSCOdd",&track.propagatedSimHitEtaCSCOdd);
  track_tree->Branch("propagatedSimHitPhiCSCOdd",&track.propagatedSimHitPhiCSCOdd);

  track_tree->Branch("meanSimHitRhoGEMl1Both",&track. meanSimHitRhoGEMl1Both);
  track_tree->Branch("meanSimHitEtaGEMl1Both",&track.meanSimHitEtaGEMl1Both);
  track_tree->Branch("meanSimHitPhiGEMl1Both",&track.meanSimHitPhiGEMl1Both);
  track_tree->Branch("meanSimHitRhoGEMl2Both",&track. meanSimHitRhoGEMl2Both);
  track_tree->Branch("meanSimHitEtaGEMl2Both",&track.meanSimHitEtaGEMl2Both);
  track_tree->Branch("meanSimHitPhiGEMl2Both",&track.meanSimHitPhiGEMl2Both);
  track_tree->Branch("meanSimHitRhoCSCBoth",&track. meanSimHitRhoCSCBoth);
  track_tree->Branch("meanSimHitEtaCSCBoth",&track.meanSimHitEtaCSCBoth);
  track_tree->Branch("meanSimHitPhiCSCBoth",&track.meanSimHitPhiCSCBoth);
  track_tree->Branch("propagatedSimHitRhoGEMl1Both",&track. propagatedSimHitRhoGEMl1Both);
  track_tree->Branch("propagatedSimHitEtaGEMl1Both",&track.propagatedSimHitEtaGEMl1Both);
  track_tree->Branch("propagatedSimHitPhiGEMl1Both",&track.propagatedSimHitPhiGEMl1Both);
  track_tree->Branch("propagatedSimHitRhoGEMl2Both",&track. propagatedSimHitRhoGEMl2Both);
  track_tree->Branch("propagatedSimHitEtaGEMl2Both",&track.propagatedSimHitEtaGEMl2Both);
  track_tree->Branch("propagatedSimHitPhiGEMl2Both",&track.propagatedSimHitPhiGEMl2Both);
  track_tree->Branch("propagatedSimHitRhoCSCBoth",&track. propagatedSimHitRhoCSCBoth);
  track_tree->Branch("propagatedSimHitEtaCSCBoth",&track.propagatedSimHitEtaCSCBoth);
  track_tree->Branch("propagatedSimHitPhiCSCBoth",&track.propagatedSimHitPhiCSCBoth);

  track_tree->Branch("charge",&track.charge);
  track_tree->Branch("pt",&track.pt);
  track_tree->Branch("eta",&track.eta);
  track_tree->Branch("phi",&track.phi);
  track_tree->Branch("hasGEMl1",&track.hasGEMl1);
  track_tree->Branch("hasGEMl2",&track.hasGEMl2);
  track_tree->Branch("hasCSC",&track.hasCSC);
}


void GEMSimHitAnalyzer::analyzeCSC( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = CSCHits->begin(); itHit != CSCHits->end(); ++itHit)
  {
    CSCDetId id(itHit->detUnitId());
    if (id.station() != 1) continue; // here we care only about station 1
    
    csc_sh.eventNumber = iEvent.id().event();
    csc_sh.detUnitId = itHit->detUnitId();
    csc_sh.particleType = itHit->particleType();
    csc_sh.x = itHit->localPosition().x();
    csc_sh.y = itHit->localPosition().y();
    csc_sh.energyLoss = itHit->energyLoss();
    csc_sh.pabs = itHit->pabs();
    csc_sh.timeOfFlight = itHit->timeOfFlight();

    csc_sh.endcap = id.endcap();
    csc_sh.ring = id.ring();
    csc_sh.station = id.station();
    csc_sh.chamber = id.chamber();
    csc_sh.layer = id.layer();

    LocalPoint p0(0., 0., 0.);
    GlobalPoint Gp0 = csc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(p0);
    csc_sh.Phi_0 = Gp0.phi();
    csc_sh.R_0 = Gp0.perp();

//    if(id.region()*pow(-1,id.chamber()) == 1) gem_sh.DeltaPhi = atan(-(itHit->localPosition().x())/(Gp0.perp() + itHit->localPosition().y()));
//    if(id.region()*pow(-1,id.chamber()) == -1) gem_sh.DeltaPhi = atan(itHit->localPosition().x()/(Gp0.perp() + itHit->localPosition().y()));
//    csc_sh.DeltaPhi = atan(-(itHit->localPosition().x())/(Gp0.perp() + itHit->localPosition().y()));
    if(id.endcap()==1) csc_sh.DeltaPhi = atan(itHit->localPosition().x()/(Gp0.perp() + itHit->localPosition().y()));
    if(id.endcap()==2) csc_sh.DeltaPhi = atan(-(itHit->localPosition().x())/(Gp0.perp() + itHit->localPosition().y()));
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = csc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
    
    csc_sh.globalR = hitGP.perp();
    csc_sh.globalEta = hitGP.eta();
    csc_sh.globalPhi = hitGP.phi();
    csc_sh.globalX = hitGP.x();
    csc_sh.globalY = hitGP.y();
    csc_sh.globalZ = hitGP.z();
    csc_sh_tree->Fill();

    simTrkProcessor->addSimHit(*itHit, hitGP);
  }
}


void GEMSimHitAnalyzer::analyzeRPC( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = RPCHits->begin(); itHit != RPCHits->end(); ++itHit)
  {
    RPCDetId id(itHit->detUnitId());
    if (id.region() == 0) continue; // we don't care about barrel RPCs

    rpc_sh.eventNumber = iEvent.id().event();
    rpc_sh.detUnitId = itHit->detUnitId();
    rpc_sh.particleType = itHit->particleType();
    rpc_sh.x = itHit->localPosition().x();
    rpc_sh.y = itHit->localPosition().y();
    rpc_sh.energyLoss = itHit->energyLoss();
    rpc_sh.pabs = itHit->pabs();
    rpc_sh.timeOfFlight = itHit->timeOfFlight();

    rpc_sh.region = id.region();
    rpc_sh.ring = id.ring();
    rpc_sh.station = id.station();
    rpc_sh.sector = id.sector();
    rpc_sh.layer = id.layer();
    rpc_sh.subsector = id.subsector();
    rpc_sh.roll = id.roll();
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = rpc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);

    rpc_sh.globalR = hitGP.perp();
    rpc_sh.globalEta = hitGP.eta();
    rpc_sh.globalPhi = hitGP.phi();
    rpc_sh.globalX = hitGP.x();
    rpc_sh.globalY = hitGP.y();
    rpc_sh.globalZ = hitGP.z();
    
    rpc_sh.strip=rpc_geometry->roll(itHit->detUnitId())->strip(hitLP);

    rpc_sh_tree->Fill();
  }
}

void GEMSimHitAnalyzer::analyzeGEM( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit)
  {

    gem_sh.eventNumber = iEvent.id().event();
    gem_sh.detUnitId = itHit->detUnitId();
    gem_sh.particleType = itHit->particleType();
    gem_sh.x = itHit->localPosition().x();
    gem_sh.y = itHit->localPosition().y();
    gem_sh.energyLoss = itHit->energyLoss();
    gem_sh.pabs = itHit->pabs();
    gem_sh.timeOfFlight = itHit->timeOfFlight();
    
    GEMDetId id(itHit->detUnitId());
    
    gem_sh.region = id.region();
    gem_sh.ring = id.ring();
    gem_sh.station = id.station();
    gem_sh.layer = id.layer();
    gem_sh.chamber = id.chamber();
    gem_sh.roll = id.roll();

    LocalPoint p0(0., 0., 0.);
    GlobalPoint Gp0 = gem_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(p0);
    gem_sh.Phi_0 = Gp0.phi();
    gem_sh.R_0 = Gp0.perp();

    if(id.region()*pow(-1,id.chamber()) == 1) gem_sh.DeltaPhi = atan(-(itHit->localPosition().x())/(Gp0.perp() + itHit->localPosition().y()));
    if(id.region()*pow(-1,id.chamber()) == -1) gem_sh.DeltaPhi = atan(itHit->localPosition().x()/(Gp0.perp() + itHit->localPosition().y()));
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = gem_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
    
    gem_sh.globalR = hitGP.perp();
    gem_sh.globalEta = hitGP.eta();
    gem_sh.globalPhi = hitGP.phi();
    gem_sh.globalX = hitGP.x();
    gem_sh.globalY = hitGP.y();
    gem_sh.globalZ = hitGP.z();

//  Now filling strip info using entry point rather than local position to be
//  consistent with digi strips. To change back, just switch the comments - WHF
//    gem_sh.strip=gem_geometry->etaPartition(itHit->detUnitId())->strip(hitLP);
    LocalPoint hitEP = itHit->entryPoint();
    gem_sh.strip=gem_geometry->etaPartition(itHit->detUnitId())->strip(hitEP);

    gem_sh_tree->Fill();

    simTrkProcessor->addSimHit(*itHit, hitGP);
  }
}


void GEMSimHitAnalyzer::analyzeTracks()
{
  GlobalPoint p0; // "point zero"

  for(size_t itrk = 0; itrk < simTrkProcessor->size(); ++itrk)
  {
    GEMSimTracksProcessor::ChamberType has_gem_l1 = simTrkProcessor->chamberTypesHitGEM(itrk, 1);
    GEMSimTracksProcessor::ChamberType has_gem_l2 = simTrkProcessor->chamberTypesHitGEM(itrk, 2);
    GEMSimTracksProcessor::ChamberType has_csc    = simTrkProcessor->chamberTypesHitCSC(itrk);

    // we want only to look at tracks that have hits in both GEM & CSC
    if ( !( has_csc && (has_gem_l1 || has_gem_l2) ) ) continue;

    cout<<"======== simtrack "<<itrk<<" ========="<<endl;
    cout<<"has gem1 gem2 csc "<<has_gem_l1<<" "<<has_gem_l2<<" "<<has_csc<<endl;

    std::set<uint32_t> gem_ids =      simTrkProcessor->getDetIdsGEM(itrk, GEMSimTracksProcessor::BOTH);
    std::set<uint32_t> csc_ids =      simTrkProcessor->getDetIdsCSC(itrk, GEMSimTracksProcessor::BOTH);
    std::set<uint32_t> gem_ids_odd =  simTrkProcessor->getDetIdsGEM(itrk, GEMSimTracksProcessor::ODD);
    std::set<uint32_t> csc_ids_odd =  simTrkProcessor->getDetIdsCSC(itrk, GEMSimTracksProcessor::ODD);
    std::set<uint32_t> gem_ids_even = simTrkProcessor->getDetIdsGEM(itrk, GEMSimTracksProcessor::EVEN);
    std::set<uint32_t> csc_ids_even = simTrkProcessor->getDetIdsCSC(itrk, GEMSimTracksProcessor::EVEN);

    cout<<"#detids: gem "<<gem_ids.size()<<" = "<<gem_ids_odd.size()<<"+"<<gem_ids_even.size()<<"   csc "<<csc_ids.size()<<" = "<<csc_ids_odd.size()<<"+"<<csc_ids_even.size()<<endl;

    // Important note: the global point is given by (0,0,0) by default when the track does not leave any SimHits in GEM layer 1. Exclusion of these points is done by requiring rho>0
    GlobalPoint sh_gem_l1_even =  simTrkProcessor->meanSimHitsPositionGEM(itrk, 1, GEMSimTracksProcessor::EVEN);
    GlobalPoint sh_gem_l2_even =  simTrkProcessor->meanSimHitsPositionGEM(itrk, 2, GEMSimTracksProcessor::EVEN);
    GlobalPoint sh_csc_even =     simTrkProcessor->meanSimHitsPositionCSC(itrk,    GEMSimTracksProcessor::EVEN);
    GlobalPoint trk_gem_l1_even = simTrkProcessor->propagatedPositionGEM(itrk, 1,  GEMSimTracksProcessor::EVEN);
    GlobalPoint trk_gem_l2_even = simTrkProcessor->propagatedPositionGEM(itrk, 2,  GEMSimTracksProcessor::EVEN);
    GlobalPoint trk_csc_even =    simTrkProcessor->propagatedPositionCSC(itrk,     GEMSimTracksProcessor::EVEN);

    GlobalPoint sh_gem_l1_odd =  simTrkProcessor->meanSimHitsPositionGEM(itrk, 1, GEMSimTracksProcessor::ODD);
    GlobalPoint sh_gem_l2_odd =  simTrkProcessor->meanSimHitsPositionGEM(itrk, 2, GEMSimTracksProcessor::ODD);
    GlobalPoint sh_csc_odd =     simTrkProcessor->meanSimHitsPositionCSC(itrk,    GEMSimTracksProcessor::ODD);
    GlobalPoint trk_gem_l1_odd = simTrkProcessor->propagatedPositionGEM(itrk, 1,  GEMSimTracksProcessor::ODD);
    GlobalPoint trk_gem_l2_odd = simTrkProcessor->propagatedPositionGEM(itrk, 2,  GEMSimTracksProcessor::ODD);
    GlobalPoint trk_csc_odd =    simTrkProcessor->propagatedPositionCSC(itrk,     GEMSimTracksProcessor::ODD);

    GlobalPoint sh_gem_l1_both =  simTrkProcessor->meanSimHitsPositionGEM(itrk, 1, GEMSimTracksProcessor::BOTH);
    GlobalPoint sh_gem_l2_both =  simTrkProcessor->meanSimHitsPositionGEM(itrk, 2, GEMSimTracksProcessor::BOTH);
    GlobalPoint sh_csc_both =     simTrkProcessor->meanSimHitsPositionCSC(itrk,    GEMSimTracksProcessor::BOTH);
    GlobalPoint trk_gem_l1_both = simTrkProcessor->propagatedPositionGEM(itrk, 1,  GEMSimTracksProcessor::BOTH);
    GlobalPoint trk_gem_l2_both = simTrkProcessor->propagatedPositionGEM(itrk, 2,  GEMSimTracksProcessor::BOTH);
    GlobalPoint trk_csc_both =    simTrkProcessor->propagatedPositionCSC(itrk,     GEMSimTracksProcessor::BOTH);

    track.meanSimHitRhoGEMl1Even = sh_gem_l1_even.perp();
    track.meanSimHitEtaGEMl1Even = sh_gem_l1_even.eta();
    track.meanSimHitPhiGEMl1Even = sh_gem_l1_even.phi();
    track.meanSimHitRhoGEMl2Even = sh_gem_l2_even.perp();
    track.meanSimHitEtaGEMl2Even = sh_gem_l2_even.eta();
    track.meanSimHitPhiGEMl2Even = sh_gem_l2_even.phi();
    track.meanSimHitRhoCSCEven = sh_csc_even.perp();
    track.meanSimHitEtaCSCEven = sh_csc_even.eta();
    track.meanSimHitPhiCSCEven = sh_csc_even.phi();
    track.propagatedSimHitRhoGEMl1Even = trk_gem_l1_even.perp();
    track.propagatedSimHitEtaGEMl1Even = trk_gem_l1_even.eta();
    track.propagatedSimHitPhiGEMl1Even = trk_gem_l1_even.phi();
    track.propagatedSimHitRhoGEMl2Even = trk_gem_l2_even.perp();
    track.propagatedSimHitEtaGEMl2Even = trk_gem_l2_even.eta();
    track.propagatedSimHitPhiGEMl2Even = trk_gem_l2_even.phi();
    track.propagatedSimHitRhoCSCEven = trk_csc_even.perp();
    track.propagatedSimHitEtaCSCEven = trk_csc_even.eta();
    track.propagatedSimHitPhiCSCEven = trk_csc_even.phi();

    track.meanSimHitRhoGEMl1Odd = sh_gem_l1_odd.perp();
    track.meanSimHitEtaGEMl1Odd = sh_gem_l1_odd.eta();
    track.meanSimHitPhiGEMl1Odd = sh_gem_l1_odd.phi();
    track.meanSimHitRhoGEMl2Odd = sh_gem_l2_odd.perp();
    track.meanSimHitEtaGEMl2Odd = sh_gem_l2_odd.eta();
    track.meanSimHitPhiGEMl2Odd = sh_gem_l2_odd.phi();
    track.meanSimHitRhoCSCOdd = sh_csc_odd.perp();
    track.meanSimHitEtaCSCOdd = sh_csc_odd.eta();
    track.meanSimHitPhiCSCOdd = sh_csc_odd.phi();
    track.propagatedSimHitRhoGEMl1Odd = trk_gem_l1_odd.perp();
    track.propagatedSimHitEtaGEMl1Odd = trk_gem_l1_odd.eta();
    track.propagatedSimHitPhiGEMl1Odd = trk_gem_l1_odd.phi();
    track.propagatedSimHitRhoGEMl2Odd = trk_gem_l2_odd.perp();
    track.propagatedSimHitEtaGEMl2Odd = trk_gem_l2_odd.eta();
    track.propagatedSimHitPhiGEMl2Odd = trk_gem_l2_odd.phi();
    track.propagatedSimHitRhoCSCOdd = trk_csc_odd.perp();
    track.propagatedSimHitEtaCSCOdd = trk_csc_odd.eta();
    track.propagatedSimHitPhiCSCOdd = trk_csc_odd.phi();

    track.meanSimHitRhoGEMl1Both = sh_gem_l1_both.perp();
    track.meanSimHitEtaGEMl1Both = sh_gem_l1_both.eta();
    track.meanSimHitPhiGEMl1Both = sh_gem_l1_both.phi();
    track.meanSimHitRhoGEMl2Both = sh_gem_l2_both.perp();
    track.meanSimHitEtaGEMl2Both = sh_gem_l2_both.eta();
    track.meanSimHitPhiGEMl2Both = sh_gem_l2_both.phi();
    track.meanSimHitRhoCSCBoth = sh_csc_both.perp();
    track.meanSimHitEtaCSCBoth = sh_csc_both.eta();
    track.meanSimHitPhiCSCBoth = sh_csc_both.phi();
    track.propagatedSimHitRhoGEMl1Both = trk_gem_l1_both.perp();
    track.propagatedSimHitEtaGEMl1Both = trk_gem_l1_both.eta();
    track.propagatedSimHitPhiGEMl1Both = trk_gem_l1_both.phi();
    track.propagatedSimHitRhoGEMl2Both = trk_gem_l2_both.perp();
    track.propagatedSimHitEtaGEMl2Both = trk_gem_l2_both.eta();
    track.propagatedSimHitPhiGEMl2Both = trk_gem_l2_both.phi();
    track.propagatedSimHitRhoCSCBoth = trk_csc_both.perp();
    track.propagatedSimHitEtaCSCBoth = trk_csc_both.eta();
    track.propagatedSimHitPhiCSCBoth = trk_csc_both.phi();

    track.charge = simTrkProcessor->track(itrk)->charge();
    track.pt = simTrkProcessor->track(itrk)->trackerSurfaceMomentum().Pt();
    track.eta = simTrkProcessor->track(itrk)->trackerSurfaceMomentum().Eta();
    track.phi = simTrkProcessor->track(itrk)->trackerSurfaceMomentum().Phi();
    track.hasGEMl1 = static_cast<int>(has_gem_l1);
    track.hasGEMl2 = static_cast<int>(has_gem_l2);
    track.hasCSC = static_cast<int>(has_csc);
    
    cout<<"========="<<endl;

    track_tree->Fill();    
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GEMSimHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(GEMSimHitAnalyzer);
