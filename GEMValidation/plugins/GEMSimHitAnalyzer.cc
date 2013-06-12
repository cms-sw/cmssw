// -*- C++ -*-
//
// Package:    GEMSimHitAnalyzer
// Class:      GEMSimHitAnalyzer
// 
// \class GEMSimHitAnalyzer
//
// Description: Analyzer GEM SimHit information
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
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "GEMCode/GEMValidation/src/SimTrackMatchManager.h"

#include "TTree.h"


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
  
  void bookGEMSimHitsTree();
  void bookSimTracksTree();
    
  void analyzeGEM( const edm::Event& iEvent );
  bool isSimTrackGood(const SimTrack &t);
  void analyzeTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  
  TTree* gem_sh_tree_;
  TTree* track_tree_;
  
  edm::Handle<edm::PSimHitContainer> GEMHits;
  edm::Handle<edm::SimTrackContainer> simTracks;
  edm::Handle<edm::SimVertexContainer> simVertices;
  
  edm::ESHandle<GEMGeometry> gem_geom;
  
  const GEMGeometry* gem_geometry;
  
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
  iSetup.get<MuonGeometryRecord>().get(gem_geom);
  gem_geometry = &*gem_geom;

  iEvent.getByLabel(simInputLabel_, simTracks);
  iEvent.getByLabel(simInputLabel_, simVertices);

  iEvent.getByLabel(edm::InputTag(simInputLabel_,"MuonGEMHits"), GEMHits);
  if(GEMHits->size()) analyzeGEM( iEvent );  
 
  analyzeTracks(iEvent,iSetup);
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
    //  gem_sh.strip=gem_geometry->etaPartition(itHit->detUnitId())->strip(hitLP);
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
