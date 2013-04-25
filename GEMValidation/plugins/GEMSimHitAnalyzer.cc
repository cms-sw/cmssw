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

#include "RPCGEM/GEMValidation/src/SimTrackMatchManager.h"

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
  Float_t charge, pt, eta, phi, x, y;
  Char_t endcap;
  Char_t gem_sh_layer1, gem_sh_layer2; // bit1: in odd  bit2: even
  Float_t gem_sh_eta, gem_sh_phi;
  Float_t gem_sh_x, gem_sh_y;
  Float_t csc_sh_eta, csc_sh_phi;
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
  bool isSimTrackGood(const SimTrack &t);
  void analyzeTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  
  TTree* csc_sh_tree_;
  TTree* rpc_sh_tree_;
  TTree* gem_sh_tree_;
  TTree* track_tree_;
  
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
 
  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  float minPt_;
  int verbose_;
};

// Constructor
GEMSimHitAnalyzer::GEMSimHitAnalyzer(const edm::ParameterSet& ps)
: cfg_(ps.getParameterSet("simTrackMatching"))
, simInputLabel_(ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits"))
, minPt_(ps.getUntrackedParameter<double>("minPt", 4.5))
, verbose_(ps.getUntrackedParameter<int>("verbose", 0))
{
  bookCSCSimHitsTree();
  bookRPCSimHitsTree();
  bookGEMSimHitsTree();  
  bookSimTracksTree();
}


GEMSimHitAnalyzer::~GEMSimHitAnalyzer()
{
}


void GEMSimHitAnalyzer::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
{
}


void GEMSimHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(csc_geom);
  csc_geometry = &*csc_geom;
  
  iSetup.get<MuonGeometryRecord>().get(rpc_geom);
  rpc_geometry = &*rpc_geom;
  
  iSetup.get<MuonGeometryRecord>().get(gem_geom);
  gem_geometry = &*gem_geom;

  iEvent.getByLabel(simInputLabel_, simTracks);
  iEvent.getByLabel(simInputLabel_, simVertices);

  iEvent.getByLabel(edm::InputTag(simInputLabel_,"MuonCSCHits"), CSCHits);
  if(CSCHits->size()) analyzeCSC( iEvent );
  
  iEvent.getByLabel(edm::InputTag(simInputLabel_,"MuonRPCHits"), RPCHits);
  if(RPCHits->size()) analyzeRPC( iEvent );
  
  iEvent.getByLabel(edm::InputTag(simInputLabel_,"MuonGEMHits"), GEMHits);
  if(GEMHits->size()) analyzeGEM( iEvent );  
 
  analyzeTracks(iEvent,iSetup);
}


void GEMSimHitAnalyzer::bookCSCSimHitsTree()
{  
  edm::Service<TFileService> fs;
  csc_sh_tree_ = fs->make<TTree>("CSCSimHits", "CSCSimHits");
  csc_sh_tree_->Branch("eventNumber",&csc_sh.eventNumber);
  csc_sh_tree_->Branch("detUnitId",&csc_sh.detUnitId);
  csc_sh_tree_->Branch("particleType",&csc_sh.particleType);
  csc_sh_tree_->Branch("x",&csc_sh.x);
  csc_sh_tree_->Branch("y",&csc_sh.y);
  csc_sh_tree_->Branch("energyLoss",&csc_sh.energyLoss);
  csc_sh_tree_->Branch("pabs",&csc_sh.pabs);
  csc_sh_tree_->Branch("timeOfFlight",&csc_sh.timeOfFlight);
  csc_sh_tree_->Branch("timeOfFlight",&csc_sh.timeOfFlight);
  csc_sh_tree_->Branch("endcap",&csc_sh.endcap);
  csc_sh_tree_->Branch("ring",&csc_sh.ring);
  csc_sh_tree_->Branch("station",&csc_sh.station);
  csc_sh_tree_->Branch("chamber",&csc_sh.chamber);
  csc_sh_tree_->Branch("layer",&csc_sh.layer);
  csc_sh_tree_->Branch("globalR",&csc_sh.globalR);
  csc_sh_tree_->Branch("globalEta",&csc_sh.globalEta);
  csc_sh_tree_->Branch("globalPhi",&csc_sh.globalPhi);
  csc_sh_tree_->Branch("globalX",&csc_sh.globalX);
  csc_sh_tree_->Branch("globalY",&csc_sh.globalY);
  csc_sh_tree_->Branch("globalZ",&csc_sh.globalZ);
  csc_sh_tree_->Branch("Phi_0", &csc_sh.Phi_0);
  csc_sh_tree_->Branch("DeltaPhi", &csc_sh.DeltaPhi);
  csc_sh_tree_->Branch("R_0", &csc_sh.R_0);
}


void GEMSimHitAnalyzer::bookRPCSimHitsTree()
{
  edm::Service< TFileService > fs;
  rpc_sh_tree_ = fs->make< TTree >("RPCSimHits", "RPCSimHits");
  rpc_sh_tree_->Branch("eventNumber", &rpc_sh.eventNumber);
  rpc_sh_tree_->Branch("detUnitId", &rpc_sh.detUnitId);
  rpc_sh_tree_->Branch("particleType", &rpc_sh.particleType);
  rpc_sh_tree_->Branch("x", &rpc_sh.x);
  rpc_sh_tree_->Branch("y", &rpc_sh.y);
  rpc_sh_tree_->Branch("energyLoss", &rpc_sh.energyLoss);
  rpc_sh_tree_->Branch("pabs", &rpc_sh.pabs);
  rpc_sh_tree_->Branch("timeOfFlight", &rpc_sh.timeOfFlight);
  rpc_sh_tree_->Branch("timeOfFlight", &rpc_sh.timeOfFlight);
  rpc_sh_tree_->Branch("ring", &rpc_sh.ring);
  rpc_sh_tree_->Branch("station", &rpc_sh.station);
  rpc_sh_tree_->Branch("layer", &rpc_sh.layer);
  rpc_sh_tree_->Branch("globalR", &rpc_sh.globalR);
  rpc_sh_tree_->Branch("globalEta", &rpc_sh.globalEta);
  rpc_sh_tree_->Branch("globalPhi", &rpc_sh.globalPhi);
  rpc_sh_tree_->Branch("globalX", &rpc_sh.globalX);
  rpc_sh_tree_->Branch("globalY", &rpc_sh.globalY);
  rpc_sh_tree_->Branch("globalZ", &rpc_sh.globalZ);
  rpc_sh_tree_->Branch("strip", &rpc_sh.strip);
}


void GEMSimHitAnalyzer::bookGEMSimHitsTree()
{
  edm::Service< TFileService > fs;
  gem_sh_tree_ = fs->make< TTree >("GEMSimHits", "GEMSimHits");
  gem_sh_tree_->Branch("eventNumber", &gem_sh.eventNumber);
  gem_sh_tree_->Branch("detUnitId", &gem_sh.detUnitId);
  gem_sh_tree_->Branch("particleType", &gem_sh.particleType);
  gem_sh_tree_->Branch("x", &gem_sh.x);
  gem_sh_tree_->Branch("y", &gem_sh.y);
  gem_sh_tree_->Branch("energyLoss", &gem_sh.energyLoss);
  gem_sh_tree_->Branch("pabs", &gem_sh.pabs);
  gem_sh_tree_->Branch("timeOfFlight", &gem_sh.timeOfFlight);
  gem_sh_tree_->Branch("region", &gem_sh.region);
  gem_sh_tree_->Branch("ring", &gem_sh.ring);
  gem_sh_tree_->Branch("station", &gem_sh.station);
  gem_sh_tree_->Branch("chamber", &gem_sh.chamber);
  gem_sh_tree_->Branch("layer", &gem_sh.layer);
  gem_sh_tree_->Branch("roll", &gem_sh.roll);
  gem_sh_tree_->Branch("globalR", &gem_sh.globalR);
  gem_sh_tree_->Branch("globalEta", &gem_sh.globalEta);
  gem_sh_tree_->Branch("globalPhi", &gem_sh.globalPhi);
  gem_sh_tree_->Branch("globalX", &gem_sh.globalX);
  gem_sh_tree_->Branch("globalY", &gem_sh.globalY);
  gem_sh_tree_->Branch("globalZ", &gem_sh.globalZ);
  gem_sh_tree_->Branch("strip", &gem_sh.strip);
  gem_sh_tree_->Branch("Phi_0", &gem_sh.Phi_0);
  gem_sh_tree_->Branch("DeltaPhi", &gem_sh.DeltaPhi);
  gem_sh_tree_->Branch("R_0", &gem_sh.R_0);
}


void GEMSimHitAnalyzer::bookSimTracksTree()
{
  edm::Service< TFileService > fs;
  track_tree_ = fs->make< TTree >("Tracks", "Tracks");
  track_tree_->Branch("charge",&track.charge);
  track_tree_->Branch("pt",&track.pt);
  track_tree_->Branch("eta",&track.eta);
  track_tree_->Branch("phi",&track.phi);
  track_tree_->Branch("x", &track.x);
  track_tree_->Branch("y", &track.y);
  track_tree_->Branch("endcap",&track.endcap);
  track_tree_->Branch("gem_sh_layer1",&track.gem_sh_layer1);
  track_tree_->Branch("gem_sh_layer2",&track.gem_sh_layer2);
  track_tree_->Branch("gem_sh_eta",&track.gem_sh_eta);
  track_tree_->Branch("gem_sh_phi",&track.gem_sh_phi);
  track_tree_->Branch("gem_sh_x",&track.gem_sh_x);
  track_tree_->Branch("gem_sh_y",&track.gem_sh_y);
  track_tree_->Branch("csc_sh_eta",&track.csc_sh_eta);
  track_tree_->Branch("csc_sh_phi",&track.csc_sh_phi);
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
    csc_sh_tree_->Fill();

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

    rpc_sh_tree_->Fill();
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

    gem_sh_tree_->Fill();
  }
}

bool GEMSimHitAnalyzer::isSimTrackGood(const SimTrack &t)
{
  // SimTrack selection
  if (t.noVertex()) return false;
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < minPt_) return false;
  float eta = std::abs(t.momentum().eta());
  if (eta > 2.18 || eta < 1.55) return false; // no GEMs could be in such eta
  return true;
}

void GEMSimHitAnalyzer::analyzeTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const edm::SimVertexContainer & sim_vert = *simVertices.product();
  
  for (auto& t: *simTracks.product())
  {
    if (!isSimTrackGood(t)) continue;
    
    // match hits and digis to this SimTrack
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup);
    
    const SimHitMatcher& match_sh = match.simhits();
    //    const SimTrack &t = match_sh.trk();
   
    track.pt = t.momentum().pt();
    track.phi = t.momentum().phi();
    track.eta = t.momentum().eta();
    track.x = -999.;
    track.y = 999.;
    track.charge = t.charge();
    track.endcap = (track.eta > 0.) ? 1 : -1;
    track.gem_sh_layer1 = 0;
    track.gem_sh_layer2 = 0;
    track.gem_sh_eta = -9.;
    track.gem_sh_phi = -9.;
    track.gem_sh_x = -999.;
    track.gem_sh_y = -999.;
    track.csc_sh_eta = -9.;
    track.csc_sh_phi = -9.;

    // ** CSC SimHits ** //    
    auto csc_sh_ids    = match_sh.detIdsCSC();
    auto csc_sh_ids_ch = match_sh.chamberIdsCSC();
    
    for(auto d: csc_sh_ids_ch)
    {
      CSCDetId id(d);
      int nlayers = match_sh.nLayersWithHitsInSuperChamber(d);
      if (nlayers < 4) continue;

      auto csc_simhits = match_sh.hitsInChamber(d);
      auto csc_simhits_gp = match_sh.simHitsMeanPosition(csc_simhits);

      track.csc_sh_eta = csc_simhits_gp.eta();
      track.csc_sh_phi = csc_simhits_gp.phi();
    }
    
    // ** GEM SimHits ** //    
    auto gem_sh_ids_sch = match_sh.superChamberIdsGEM();
    for(auto d: gem_sh_ids_sch)
    {
      auto gem_simhits = match_sh.hitsInSuperChamber(d);
      auto gem_simhits_gp = match_sh.simHitsMeanPosition(gem_simhits);

      track.gem_sh_eta = gem_simhits_gp.eta();
      track.gem_sh_phi = gem_simhits_gp.phi();
      track.gem_sh_x = gem_simhits_gp.x();
      track.gem_sh_y = gem_simhits_gp.y();
    }
    
    auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      GEMDetId id(d);
      bool odd = id.chamber() & 1;

      if (id.layer() == 1)
      {
        if (odd) track.gem_sh_layer1 |= 1;
        else     track.gem_sh_layer1 |= 2;
      }
      else if (id.layer() == 2)
      {
        if (odd) track.gem_sh_layer2 |= 1;
        else     track.gem_sh_layer2 |= 2;
      }
    }
    track_tree_->Fill();    
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
