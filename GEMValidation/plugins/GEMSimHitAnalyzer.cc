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
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"

struct MyCSCSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t endcap, ring, station, chamber, layer;
  Float_t globalPerp, globalEta, globalPhi, globalX, globalY, globalZ;
};


struct MyRPCSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t region, ring, station, sector, layer, subsector, roll;
  Float_t globalPerp, globalEta, globalPhi, globalX, globalY, globalZ;
};


struct MyGEMSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t region, ring, station, layer, chamber, roll;
  Float_t globalPerp, globalEta, globalPhi, globalX, globalY, globalZ;
};

struct MySimTrack
{
  Int_t particleId, trackId;
  Float_t globalX, globalY, globalZ, localX, localY, localZ;
  Float_t globalPerp, globalEta, globalPhi, localPerp, localEta, localPhi;
};


class GEMSimHitAnalyzer : public edm::EDAnalyzer
{
public:
  /// Constructor
  explicit GEMSimHitAnalyzer(const edm::ParameterSet&);
  /// Destructor
  ~GEMSimHitAnalyzer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
private:
  
  void bookCSCSimHitsTree();
  void bookRPCSimHitsTree();
  void bookGEMSimHitsTree();
  void bookSimTracksTree();
    
  void analyzeCSC( const edm::Event& iEvent );
  void analyzeRPC( const edm::Event& iEvent );
  void analyzeGEM( const edm::Event& iEvent );
  
  TTree* csc_sh_tree;
  TTree* rpc_sh_tree;
  TTree* gem_sh_tree;
  TTree* track_tree;
  
  edm::Handle<edm::PSimHitContainer> CSCHits;
  edm::Handle<edm::PSimHitContainer> RPCHits;
  edm::Handle<edm::PSimHitContainer> GEMHits;
  edm::Handle<edm::SimTrackContainer> Tracks;
  
  edm::ESHandle<CSCGeometry> csc_geom;
  edm::ESHandle<RPCGeometry> rpc_geom;
  edm::ESHandle<GEMGeometry> gem_geom;
  
  const CSCGeometry* csc_geometry;
  const RPCGeometry* rpc_geometry;
  const GEMGeometry* gem_geometry;
  
  MyCSCSimHit csc_sh;
  MyRPCSimHit rpc_sh;
  MyGEMSimHit gem_sh;
  MySimTrack  tracks; 
  GlobalPoint hitGP;
 
};

// Constructor
GEMSimHitAnalyzer::GEMSimHitAnalyzer(const edm::ParameterSet& iConfig)
{
  bookCSCSimHitsTree();
  bookRPCSimHitsTree();
  bookGEMSimHitsTree();  
  bookSimTracksTree();
}

// Destructor
GEMSimHitAnalyzer::~GEMSimHitAnalyzer()
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
  
  edm::InputTag default_tag_csc("g4SimHits","MuonCSCHits");
  iEvent.getByLabel(default_tag_csc, CSCHits);
  if(CSCHits->size()) analyzeCSC( iEvent );
  
  edm::InputTag default_tag_rpc("g4SimHits","MuonRPCHits");
  iEvent.getByLabel(default_tag_rpc, RPCHits);
  if(RPCHits->size()) analyzeRPC( iEvent );
  
  edm::InputTag default_tag_gem("g4SimHits","MuonGEMHits");
  iEvent.getByLabel(default_tag_gem, GEMHits);
  if(GEMHits->size()) analyzeGEM( iEvent );  
 
  edm::InputTag default_tag_track("g4SimTracks");
  iEvent.getByLabel(default_tag_track, Tracks);

  //  for ( edm::SimTrackContainer::const_iterator iTrack = Tracks->begin(); iTrack != Tracks->end(); ++iTrack) {
    
//     tracks.particleId = iTrack.type();
//     tracks.trackId = iTrack.trackId();    
//     tracks.globalX = iTrack->trackerSurfacePosition().X();
//     tracks.globalY = iTrack->trackerSurfacePosition().Y();
//     tracks.globalZ = iTrack->trackerSurfacePosition().Z();
//     tracks.globalPerp = iTrack->trackerSurfaceMomentum().Perp();
//     tracks.globalEta = iTrack->trackerSurfaceMomentum().Eta();
//     tracks.globalPhi = iTrack->trackerSurfaceMomentum().Phi();
//     tracks.localX = iTrack->trackerSurfacePosition()->x();
//     tracks.localY = iTrack->trackerSurfacePosition()->y();
//     tracks.localZ = iTrack->trackerSurfacePosition()->z();
//     tracks.localPerp = iTrack->trackerSurfacePosition()->Y();
//     tracks.localEta = iTrack->trackerSurfacePosition()->Y();
//     tracks.localPhi = iTrack->trackerSurfacePosition()->Y();
//  }
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
  csc_sh_tree->Branch("globalPerp",&csc_sh.globalPerp);
  csc_sh_tree->Branch("globalEta",&csc_sh.globalEta);
  csc_sh_tree->Branch("globalPhi",&csc_sh.globalPhi);
  csc_sh_tree->Branch("globalX",&csc_sh.globalX);
  csc_sh_tree->Branch("globalY",&csc_sh.globalY);
  csc_sh_tree->Branch("globalZ",&csc_sh.globalZ);
}

void GEMSimHitAnalyzer::bookRPCSimHitsTree()
{
  edm::Service<TFileService> fs;
  rpc_sh_tree = fs->make<TTree>("RPCSimHits", "RPCSimHits");
  rpc_sh_tree->Branch("eventNumber",&rpc_sh.eventNumber);
  rpc_sh_tree->Branch("detUnitId",&rpc_sh.detUnitId);
  rpc_sh_tree->Branch("particleType",&rpc_sh.particleType);
  rpc_sh_tree->Branch("x",&rpc_sh.x);
  rpc_sh_tree->Branch("y",&rpc_sh.y);
  rpc_sh_tree->Branch("energyLoss",&rpc_sh.energyLoss);
  rpc_sh_tree->Branch("pabs",&rpc_sh.pabs);
  rpc_sh_tree->Branch("timeOfFlight",&rpc_sh.timeOfFlight);
  rpc_sh_tree->Branch("timeOfFlight",&rpc_sh.timeOfFlight);
  rpc_sh_tree->Branch("ring",&rpc_sh.ring);
  rpc_sh_tree->Branch("station",&rpc_sh.station);
  rpc_sh_tree->Branch("layer",&rpc_sh.layer);
  rpc_sh_tree->Branch("globalPerp",&rpc_sh.globalPerp);
  rpc_sh_tree->Branch("globalEta",&rpc_sh.globalEta);
  rpc_sh_tree->Branch("globalPhi",&rpc_sh.globalPhi);
  rpc_sh_tree->Branch("globalX",&rpc_sh.globalX);
  rpc_sh_tree->Branch("globalY",&rpc_sh.globalY);
  rpc_sh_tree->Branch("globalZ",&rpc_sh.globalZ);
}

void GEMSimHitAnalyzer::bookGEMSimHitsTree()
{  
  edm::Service<TFileService> fs;
  gem_sh_tree = fs->make<TTree>("GEMSimHits", "GEMSimHits");
  gem_sh_tree->Branch("eventNumber",&gem_sh.eventNumber);
  gem_sh_tree->Branch("detUnitId",&gem_sh.detUnitId);
  gem_sh_tree->Branch("particleType",&gem_sh.particleType);
  gem_sh_tree->Branch("x",&gem_sh.x);
  gem_sh_tree->Branch("y",&gem_sh.y);
  gem_sh_tree->Branch("energyLoss",&gem_sh.energyLoss);
  gem_sh_tree->Branch("pabs",&gem_sh.pabs);
  gem_sh_tree->Branch("timeOfFlight",&gem_sh.timeOfFlight);
  gem_sh_tree->Branch("region",&gem_sh.region);
  gem_sh_tree->Branch("ring",&gem_sh.ring);
  gem_sh_tree->Branch("station",&gem_sh.station);
  gem_sh_tree->Branch("chamber",&gem_sh.chamber);
  gem_sh_tree->Branch("layer",&gem_sh.layer);
  gem_sh_tree->Branch("globalPerp",&gem_sh.globalPerp);
  gem_sh_tree->Branch("globalEta",&gem_sh.globalEta);
  gem_sh_tree->Branch("globalPhi",&gem_sh.globalPhi);
  gem_sh_tree->Branch("globalX",&gem_sh.globalX);
  gem_sh_tree->Branch("globalY",&gem_sh.globalY);
  gem_sh_tree->Branch("globalZ",&gem_sh.globalZ);
}

void GEMSimHitAnalyzer::bookSimTracksTree()
{
  edm::Service<TFileService> fs;
  track_tree = fs->make<TTree>("Tracks", "Tracks");
  track_tree->Branch("particleId",&tracks.particleId);
  track_tree->Branch("trackId",&tracks.trackId);
  track_tree->Branch("globalX",&tracks.globalX);
  track_tree->Branch("globalY",&tracks.globalY);  
  track_tree->Branch("globalZ",&tracks.globalZ);
  track_tree->Branch("globalPerp",&tracks.globalPerp);
  track_tree->Branch("globalEta",&tracks.globalEta);
  track_tree->Branch("globalPhi",&tracks.globalPhi);
  track_tree->Branch("localX",&tracks.localX);
  track_tree->Branch("localY",&tracks.localY);
  track_tree->Branch("localZ",&tracks.localZ);
  track_tree->Branch("localPerp",&tracks.localPerp);
  track_tree->Branch("localEta",&tracks.localEta);  
  track_tree->Branch("localPhi",&tracks.localPhi);
}

void GEMSimHitAnalyzer::analyzeCSC( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = CSCHits->begin(); itHit != CSCHits->end(); ++itHit) {
    CSCDetId id(itHit->detUnitId());
    if (id.station() != 1) continue; // here we care only about station 1
    
    csc_sh.eventNumber=iEvent.id().event();
    csc_sh.detUnitId=itHit->detUnitId();
    csc_sh.particleType=itHit->particleType();
    csc_sh.x=itHit->localPosition().x();
    csc_sh.y=itHit->localPosition().y();
    csc_sh.energyLoss=itHit->energyLoss();
    csc_sh.pabs=itHit->pabs();
    csc_sh.timeOfFlight=itHit->timeOfFlight();    

    csc_sh.endcap=id.endcap();
    csc_sh.ring=id.ring();
    csc_sh.station=id.station();
    csc_sh.chamber=id.chamber();
    csc_sh.layer=id.layer();
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = csc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
    
    csc_sh.globalPerp=hitGP.perp();
    csc_sh.globalEta=hitGP.eta();
    csc_sh.globalPhi=hitGP.phi();
    csc_sh.globalX=hitGP.x();
    csc_sh.globalY=hitGP.y();
    csc_sh.globalZ=hitGP.z();
    csc_sh_tree->Fill();
  }  
}

void GEMSimHitAnalyzer::analyzeRPC( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = RPCHits->begin(); itHit != RPCHits->end(); ++itHit) {     
    RPCDetId id(itHit->detUnitId());
    if (id.region() == 0) continue; // we don't care about barrel RPCs

    csc_sh.eventNumber=iEvent.id().event();
    rpc_sh.detUnitId=itHit->detUnitId();
    rpc_sh.particleType=itHit->particleType();
    rpc_sh.x=itHit->localPosition().x();
    rpc_sh.y=itHit->localPosition().y();
    rpc_sh.energyLoss=itHit->energyLoss();
    rpc_sh.pabs=itHit->pabs();
    rpc_sh.timeOfFlight=itHit->timeOfFlight();
    
    rpc_sh.region=id.region();
    rpc_sh.ring=id.ring();
    rpc_sh.station=id.station();
    rpc_sh.sector=id.sector();
    rpc_sh.layer=id.layer();
    rpc_sh.subsector=id.subsector();
    rpc_sh.roll=id.roll();
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = rpc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
    
    rpc_sh.globalPerp=hitGP.perp();
    rpc_sh.globalEta=hitGP.eta();
    rpc_sh.globalPhi=hitGP.phi();
    rpc_sh.globalX=hitGP.x();
    rpc_sh.globalY=hitGP.y();
    rpc_sh.globalZ=hitGP.z();
    
    rpc_sh_tree->Fill();
   }
}

void GEMSimHitAnalyzer::analyzeGEM( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit) {     

    csc_sh.eventNumber=iEvent.id().event();    
    gem_sh.detUnitId=itHit->detUnitId();
    gem_sh.particleType=itHit->particleType();
    gem_sh.x=itHit->localPosition().x();
    gem_sh.y=itHit->localPosition().y();
    gem_sh.energyLoss=itHit->energyLoss();
    gem_sh.pabs=itHit->pabs();
    gem_sh.timeOfFlight=itHit->timeOfFlight();
    
    GEMDetId id(itHit->detUnitId());
    
    gem_sh.region=id.region();
    gem_sh.ring=id.ring();
    gem_sh.station=id.station();
    gem_sh.layer=id.layer();
    gem_sh.chamber=id.chamber();
    gem_sh.roll=id.roll();
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = gem_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
    
    gem_sh.globalPerp=hitGP.perp();
    gem_sh.globalEta=hitGP.eta();
    gem_sh.globalPhi=hitGP.phi();
    gem_sh.globalX=hitGP.x();
    gem_sh.globalY=hitGP.y();
    gem_sh.globalZ=hitGP.z();
    gem_sh_tree->Fill();    
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
