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
};

struct MyRPCSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t region, ring, station, sector, layer, subsector, roll;
  Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
};


struct MyGEMSimHit
{  
  Int_t eventNumber;
  Int_t detUnitId, particleType;
  Float_t x, y, energyLoss, pabs, timeOfFlight;
  Int_t region, ring, station, layer, chamber, roll;
  Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
};

struct MySimTrack
{
  Int_t particleId, trackId;
  Float_t globalX, globalY, globalZ, globalR, globalEta, globalPhi;
  Float_t  localX,  localY,  localZ,  localR,  localEta,  localPhi;
};


class GEMSimHitAnalyzer : public edm::EDAnalyzer
{
public:
  /// Constructor
  explicit GEMSimHitAnalyzer(const edm::ParameterSet&);
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

  GEMSimTracksProcessor simTrkProcessor;

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

  bookCSCSimHitsTree();
  bookRPCSimHitsTree();
  bookGEMSimHitsTree();  
  bookSimTracksTree();
}


GEMSimHitAnalyzer::~GEMSimHitAnalyzer() {}


void GEMSimHitAnalyzer::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
{
  simTrkProcessor.init(iSetup);
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

  simTrkProcessor.fillTracks( *(simTracks.product()), *(simVertices.product()) );

  iEvent.getByLabel(edm::InputTag(simInputLabel,"MuonCSCHits"), CSCHits);
  if(CSCHits->size()) analyzeCSC( iEvent );
  
  iEvent.getByLabel(edm::InputTag(simInputLabel,"MuonRPCHits"), RPCHits);
  if(RPCHits->size()) analyzeRPC( iEvent );
  
  iEvent.getByLabel(edm::InputTag(simInputLabel,"MuonGEMHits"), GEMHits);
  if(GEMHits->size()) analyzeGEM( iEvent );  
 

  if(simTrkProcessor.size()) analyzeTracks();
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
  gem_sh_tree->Branch("globalR", &gem_sh.globalR);
  gem_sh_tree->Branch("globalEta", &gem_sh.globalEta);
  gem_sh_tree->Branch("globalPhi", &gem_sh.globalPhi);
  gem_sh_tree->Branch("globalX", &gem_sh.globalX);
  gem_sh_tree->Branch("globalY", &gem_sh.globalY);
  gem_sh_tree->Branch("globalZ", &gem_sh.globalZ);
}


void GEMSimHitAnalyzer::bookSimTracksTree()
{
  edm::Service< TFileService > fs;
  track_tree = fs->make< TTree >("Tracks", "Tracks");
  track_tree->Branch("particleId", &track.particleId);
  track_tree->Branch("trackId", &track.trackId);
  track_tree->Branch("globalX", &track.globalX);
  track_tree->Branch("globalY", &track.globalY);
  track_tree->Branch("globalZ", &track.globalZ);
  track_tree->Branch("globalR", &track.globalR);
  track_tree->Branch("globalEta", &track.globalEta);
  track_tree->Branch("globalPhi", &track.globalPhi);
  track_tree->Branch("localX", &track.localX);
  track_tree->Branch("localY", &track.localY);
  track_tree->Branch("localZ", &track.localZ);
  track_tree->Branch("localR", &track.localR);
  track_tree->Branch("localEta", &track.localEta);
  track_tree->Branch("localPhi", &track.localPhi);
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
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = csc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
    
    csc_sh.globalR = hitGP.perp();
    csc_sh.globalEta = hitGP.eta();
    csc_sh.globalPhi = hitGP.phi();
    csc_sh.globalX = hitGP.x();
    csc_sh.globalY = hitGP.y();
    csc_sh.globalZ = hitGP.z();
    csc_sh_tree->Fill();

    simTrkProcessor.addSimHit(*itHit, hitGP);
  }
}


void GEMSimHitAnalyzer::analyzeRPC( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = RPCHits->begin(); itHit != RPCHits->end(); ++itHit)
  {
    RPCDetId id(itHit->detUnitId());
    if (id.region() == 0) continue; // we don't care about barrel RPCs
    
    csc_sh.eventNumber = iEvent.id().event();
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
    
    rpc_sh_tree->Fill();
  }
}

void GEMSimHitAnalyzer::analyzeGEM( const edm::Event& iEvent )
{
  for (edm::PSimHitContainer::const_iterator itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit)
  {

    csc_sh.eventNumber = iEvent.id().event();
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
    
    LocalPoint hitLP = itHit->localPosition();
    hitGP = gem_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
    
    gem_sh.globalR = hitGP.perp();
    gem_sh.globalEta = hitGP.eta();
    gem_sh.globalPhi = hitGP.phi();
    gem_sh.globalX = hitGP.x();
    gem_sh.globalY = hitGP.y();
    gem_sh.globalZ = hitGP.z();
    gem_sh_tree->Fill();

    simTrkProcessor.addSimHit(*itHit, hitGP);
  }
}


void GEMSimHitAnalyzer::analyzeTracks()
{
  GlobalPoint p0; // "point zero"

  for(size_t itrk = 0; itrk < simTrkProcessor.size(); ++itrk)
  {
    GEMSimTracksProcessor::ChamberType has_gem_l1 = simTrkProcessor.chamberTypesHitGEM(itrk, 1);
    GEMSimTracksProcessor::ChamberType has_gem_l2 = simTrkProcessor.chamberTypesHitGEM(itrk, 2);
    GEMSimTracksProcessor::ChamberType has_csc    = simTrkProcessor.chamberTypesHitCSC(itrk);

    // want only to look at tracks that have hits in both GEM & CSC
    if ( !( has_csc && (has_gem_l1 || has_gem_l2) ) ) continue;

    cout<<"======== simtrack "<<itrk<<" ========="<<endl;
    cout<<"has gem1 gem2 csc "<<has_gem_l1<<" "<<has_gem_l2<<" "<<has_csc<<endl;

    std::set<uint32_t> gem_ids = simTrkProcessor.getDetIdsGEM(itrk, GEMSimTracksProcessor::BOTH);
    std::set<uint32_t> csc_ids = simTrkProcessor.getDetIdsCSC(itrk, GEMSimTracksProcessor::BOTH);
    std::set<uint32_t> gem_ids_odd = simTrkProcessor.getDetIdsGEM(itrk, GEMSimTracksProcessor::ODD);
    std::set<uint32_t> csc_ids_odd = simTrkProcessor.getDetIdsCSC(itrk, GEMSimTracksProcessor::ODD);
    std::set<uint32_t> gem_ids_even = simTrkProcessor.getDetIdsGEM(itrk, GEMSimTracksProcessor::EVEN);
    std::set<uint32_t> csc_ids_even = simTrkProcessor.getDetIdsCSC(itrk, GEMSimTracksProcessor::EVEN);

    cout<<"#detids: gem "<<gem_ids.size()<<" = "<<gem_ids_odd.size()<<"+"<<gem_ids_even.size()<<"   csc "<<csc_ids.size()<<" = "<<csc_ids_odd.size()<<"+"<<csc_ids_even.size()<<endl;

    GlobalPoint sh_gem_l1 = simTrkProcessor.meanSimHitsPositionGEM(itrk, 1, GEMSimTracksProcessor::BOTH);
    GlobalPoint sh_gem_l2 = simTrkProcessor.meanSimHitsPositionGEM(itrk, 2, GEMSimTracksProcessor::BOTH);
    GlobalPoint sh_csc = simTrkProcessor.meanSimHitsPositionCSC(itrk, GEMSimTracksProcessor::BOTH);

    GlobalPoint trk_gem_l1 = simTrkProcessor.propagatedPositionGEM(itrk, 1, GEMSimTracksProcessor::BOTH);
    GlobalPoint trk_gem_l2 = simTrkProcessor.propagatedPositionGEM(itrk, 2, GEMSimTracksProcessor::BOTH);
    GlobalPoint trk_csc = simTrkProcessor.propagatedPositionCSC(itrk, GEMSimTracksProcessor::BOTH);

    if (!(sh_gem_l1 == p0))
      cout<<"SH-TRK GEM L1  "<<sh_gem_l1<<" \t "<<trk_gem_l1<<" \t delta_r= "<<trk_gem_l1.perp() - sh_gem_l1.perp()
        <<" \t delta_phi*r= "<<trk_gem_l1.perp() * reco::deltaPhi<GlobalPoint,GlobalPoint>(trk_gem_l1, sh_gem_l1)<<endl;
    if (!(sh_gem_l2 == p0))
      cout<<"SH-TRK GEM L2  "<<sh_gem_l2<<" \t "<<trk_gem_l2<<" \t delta_r= "<<trk_gem_l2.perp() - sh_gem_l2.perp()
        <<" \t delta_phi*r= "<<trk_gem_l2.perp() * reco::deltaPhi<GlobalPoint,GlobalPoint>(trk_gem_l2, sh_gem_l2)<<endl;
    if (!(sh_csc == p0))
      cout<<"SH-TRK CSC     "<<sh_csc<<" \t "<<trk_csc<<" \t delta_r= "<<trk_csc.perp() - sh_csc.perp()
        <<" \t delta_phi*r= "<<trk_csc.perp() * reco::deltaPhi<GlobalPoint,GlobalPoint>(trk_csc, sh_csc)<<endl;

    if (!(sh_gem_l1 == p0 || sh_csc == p0))
      cout<<"SH CSC-GEM L1   delta_eta= "<<trk_gem_l1.eta() - sh_csc.eta()
        <<" \t delta_phi= "<<reco::deltaPhi<GlobalPoint,GlobalPoint>(trk_gem_l1, sh_csc)<<endl;
    if (!(sh_gem_l2 == p0 || sh_csc == p0))
      cout<<"SH CSC-GEM L2   delta_eta= "<<trk_gem_l2.eta() - sh_csc.eta()
        <<" \t delta_phi= "<<reco::deltaPhi<GlobalPoint,GlobalPoint>(trk_gem_l2, sh_csc)<<endl;

    cout<<"========="<<endl;
  }

  /*
  for (edm::SimTrackContainer::const_iterator iTrack = simTracks->begin(); iTrack != simTracks->end(); ++iTrack) {
    track.particleId = iTrack->type();
    track.trackId = iTrack->trackId();
    track.globalX = iTrack->trackerSurfacePosition().X();
    track.globalY = iTrack->trackerSurfacePosition().Y();
    track.globalZ = iTrack->trackerSurfacePosition().Z();
    track.globalR = iTrack->trackerSurfaceMomentum().R();
    track.globalEta = iTrack->trackerSurfaceMomentum().Eta();
    tracks.globalPhi = iTrack->trackerSurfaceMomentum().Phi();
//     tracks.localX = iTrack.trackerSurfacePosition().x();
//     tracks.localY = iTrack.trackerSurfacePosition().y();
//     tracks.localZ = iTrack.trackerSurfacePosition().z();
//     tracks.localR = iTrack.trackerSurfacePosition().R();
//     tracks.localEta = iTrack.trackerSurfacePosition().Eta();
//     track.localPhi = iTrack.trackerSurfacePosition().Phi();
    track_tree->Fill();    
 }
 */

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
