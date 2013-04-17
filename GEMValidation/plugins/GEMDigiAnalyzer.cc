// -*- C++ -*-
//
// Package:    GEMDigiAnalyzer
// Class:      GEMDigiAnalyzer
// 
/**\class GEMDigiAnalyzer GEMDigiAnalyzer.cc MyAnalyzers/GEMDigiAnalyzer/src/GEMDigiAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// $Id: GEMDigiAnalyzer.cc,v 1.7 2013/03/08 22:16:35 dildick Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TTree.h"
#include "TFile.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"


///Data Format
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "RPCGEM/GEMValidation/src/SimTrackMatchManager.h"


using namespace std;
//using namespace edm;

//
// class declaration
//

// "signed" LCT bend pattern
const int LCT_BEND_PATTERN[11] = { -99,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0};


struct MyRPCDigi
{  
   Int_t detId;
   Short_t region, ring, station, sector, layer, subsector, roll;
   Short_t strip, bx;
   Float_t x, y;
   Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MyGEMDigi
{  
   Int_t detId;
   Short_t region, ring, station, layer, chamber, roll;
   Short_t strip, bx;
   Float_t x, y;
   Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MyGEMCSCPadDigis
{
  Int_t detId;
  Short_t region, ring, station, layer, chamber, roll;
  Short_t pad, bx;
  Float_t x, y;
  Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MyGEMCSCCoPadDigis
{
  Int_t detId;
  Short_t region, ring, station, layer, chamber, roll;
  Short_t pad, bx;
  Float_t x, y;
  Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MySimTrack
{
  Float_t pt, eta, phi;
  Char_t charge;
  Char_t endcap;
  Char_t has_gem_sh; // bit1: in odd, bit2: even
  Char_t has_gem_sh2; // has SimHits in 2 layers  bit1: in odd, bit2: even
  Char_t has_gem_dg; // bit1: in odd, bit2: even
  Char_t has_gem_dg2; // has digis in 2 layers  bit1: in odd, bit2: even
  Char_t has_gem_pad; // bit1: in odd, bit2: even
  Char_t has_gem_pad2; // has pads in 2 layers  bit1: in odd, bit2: even
  Char_t has_gem_copad; // bit1: in odd, bit2: even
  Char_t gem_sh_layer; // bit1: layer1  bit2: layer2
  Char_t gem_dg_layer; // bit1: layer1  bit2: layer2
  Char_t gem_pad_layer; // bit1: layer1  bit2: layer2
  Float_t gem_sh_eta;
  Float_t gem_sh_phi;
  Float_t gem_dg_eta;
  Float_t gem_dg_phi;
  Float_t gem_pad_eta;
  Float_t gem_pad_phi;
};

class GEMDigiAnalyzer : public edm::EDAnalyzer 
{
public:
  /// constructor
  explicit GEMDigiAnalyzer(const edm::ParameterSet&);
  /// destructor
  ~GEMDigiAnalyzer();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  
  void bookRPCDigiTree();
  void bookGEMDigiTree();
  void bookGEMCSCPadDigiTree();
  void bookGEMCSCCoPadDigiTree();
  void bookSimTracksTree();

  void analyzeRPC();
  void analyzeGEM();
  void analyzeGEMCSCPad();  
  void analyzeGEMCSCCoPad();  
  bool isSimTrackGood(const SimTrack &);
  void analyzeTracks(edm::ParameterSet, const edm::Event&, const edm::EventSetup&);

  TTree* rpc_tree_;
  TTree* gem_tree_;
  TTree* gemcscpad_tree_;
  TTree* gemcsccopad_tree_;
  TTree* track_tree_;

  edm::Handle<RPCDigiCollection> rpc_digis;
  edm::Handle<GEMDigiCollection> gem_digis;  
  edm::Handle<GEMCSCPadDigiCollection> gemcscpad_digis;
  edm::Handle<GEMCSCPadDigiCollection> gemcsccopad_digis;
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;

  edm::ESHandle<RPCGeometry> rpc_geo_;
  edm::ESHandle<GEMGeometry> gem_geo_;

  edm::InputTag input_tag_rpc_;
  edm::InputTag input_tag_gem_;
  edm::InputTag input_tag_gemcscpad_;
  edm::InputTag input_tag_gemcsccopad_;
  
  MyRPCDigi rpc_digi_;
  MyGEMDigi gem_digi_;
  MyGEMCSCPadDigis gemcscpad_digi_;
  MyGEMCSCCoPadDigis gemcsccopad_digi_;
  MySimTrack track_;

  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  float minPt_;
  int verbose_;
};

//
// constructors and destructor
//
GEMDigiAnalyzer::GEMDigiAnalyzer(const edm::ParameterSet& ps)
  : cfg_(ps.getParameterSet("simTrackMatching"))
  , simInputLabel_(ps.getUntrackedParameter<std::string>("simInputLabel", "g4SimHits"))
  , minPt_(ps.getUntrackedParameter<double>("minPt", 5.))
  , verbose_(ps.getUntrackedParameter<int>("verbose", 0))
{
  bookRPCDigiTree();
  bookGEMDigiTree();
  bookGEMCSCPadDigiTree();
  bookGEMCSCCoPadDigiTree();
  bookSimTracksTree();
  edm::LogInfo("TEST") << "analyzing track" << std::endl;

}

GEMDigiAnalyzer::~GEMDigiAnalyzer() 
{
}

// ------------ method called when starting to processes a run  ------------
void GEMDigiAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(rpc_geo_);
  iSetup.get<MuonGeometryRecord>().get(gem_geo_);
}

void GEMDigiAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // FIXME - include check for digi collections
  iEvent.getByLabel(edm::InputTag("simMuonRPCDigis"), rpc_digis);
  analyzeRPC();
  
  iEvent.getByLabel(edm::InputTag("simMuonGEMDigis"), gem_digis);
  analyzeGEM();
  
  iEvent.getByLabel(edm::InputTag("simMuonGEMCSCPadDigis"), gemcscpad_digis);
  analyzeGEMCSCPad();  
  
  iEvent.getByLabel(edm::InputTag("simMuonGEMCSCPadDigis","Coincidence"), gemcsccopad_digis);
  analyzeGEMCSCCoPad();  

  iEvent.getByLabel(simInputLabel_, sim_tracks);
  iEvent.getByLabel(simInputLabel_, sim_vertices);
  analyzeTracks(cfg_,iEvent,iSetup);  

}

void GEMDigiAnalyzer::bookRPCDigiTree()
{
  edm::Service<TFileService> fs;
  rpc_tree_ = fs->make<TTree>("RPCDigiTree", "RPCDigiTree");
  rpc_tree_->Branch("detId", &rpc_digi_.detId);
  rpc_tree_->Branch("region", &rpc_digi_.region);
  rpc_tree_->Branch("ring", &rpc_digi_.ring);
  rpc_tree_->Branch("station", &rpc_digi_.station);
  rpc_tree_->Branch("sector", &rpc_digi_.sector);
  rpc_tree_->Branch("layer", &rpc_digi_.layer);
  rpc_tree_->Branch("subsector", &rpc_digi_.subsector);
  rpc_tree_->Branch("roll", &rpc_digi_.roll);
  rpc_tree_->Branch("strip", &rpc_digi_.strip);
  rpc_tree_->Branch("bx", &rpc_digi_.bx);
  rpc_tree_->Branch("x", &rpc_digi_.x);
  rpc_tree_->Branch("y", &rpc_digi_.y);
  rpc_tree_->Branch("g_r", &rpc_digi_.g_r);
  rpc_tree_->Branch("g_eta", &rpc_digi_.g_eta);
  rpc_tree_->Branch("g_phi", &rpc_digi_.g_phi);
  rpc_tree_->Branch("g_x", &rpc_digi_.g_x);
  rpc_tree_->Branch("g_y", &rpc_digi_.g_y);
  rpc_tree_->Branch("g_z", &rpc_digi_.g_z);
}  

void GEMDigiAnalyzer::bookGEMDigiTree()
{
  edm::Service<TFileService> fs;
  gem_tree_ = fs->make<TTree>("GEMDigiTree", "GEMDigiTree");
  gem_tree_->Branch("detId", &gem_digi_.detId);
  gem_tree_->Branch("region", &gem_digi_.region);
  gem_tree_->Branch("ring", &gem_digi_.ring);
  gem_tree_->Branch("station", &gem_digi_.station);
  gem_tree_->Branch("layer", &gem_digi_.layer);
  gem_tree_->Branch("chamber", &gem_digi_.chamber);
  gem_tree_->Branch("roll", &gem_digi_.roll);
  gem_tree_->Branch("strip", &gem_digi_.strip);
  gem_tree_->Branch("bx", &gem_digi_.bx);
  gem_tree_->Branch("x", &gem_digi_.x);
  gem_tree_->Branch("y", &gem_digi_.y);
  gem_tree_->Branch("g_r", &gem_digi_.g_r);
  gem_tree_->Branch("g_eta", &gem_digi_.g_eta);
  gem_tree_->Branch("g_phi", &gem_digi_.g_phi);
  gem_tree_->Branch("g_x", &gem_digi_.g_x);
  gem_tree_->Branch("g_y", &gem_digi_.g_y);
  gem_tree_->Branch("g_z", &gem_digi_.g_z);
}

void GEMDigiAnalyzer::bookGEMCSCPadDigiTree()
{
  edm::Service<TFileService> fs;
  gemcscpad_tree_ = fs->make<TTree>("GEMCSCPadDigiTree", "GEMCSCPadDigiTree");
  gemcscpad_tree_->Branch("detId", &gemcscpad_digi_.detId);
  gemcscpad_tree_->Branch("region", &gemcscpad_digi_.region);
  gemcscpad_tree_->Branch("ring", &gemcscpad_digi_.ring);
  gemcscpad_tree_->Branch("station", &gemcscpad_digi_.station);
  gemcscpad_tree_->Branch("layer", &gemcscpad_digi_.layer);
  gemcscpad_tree_->Branch("chamber", &gemcscpad_digi_.chamber);
  gemcscpad_tree_->Branch("roll", &gemcscpad_digi_.roll);
  gemcscpad_tree_->Branch("pad", &gemcscpad_digi_.pad);
  gemcscpad_tree_->Branch("bx", &gemcscpad_digi_.bx);
  gemcscpad_tree_->Branch("x", &gemcscpad_digi_.x);
  gemcscpad_tree_->Branch("y", &gemcscpad_digi_.y);
  gemcscpad_tree_->Branch("g_r", &gemcscpad_digi_.g_r);
  gemcscpad_tree_->Branch("g_eta", &gemcscpad_digi_.g_eta);
  gemcscpad_tree_->Branch("g_phi", &gemcscpad_digi_.g_phi);
  gemcscpad_tree_->Branch("g_x", &gemcscpad_digi_.g_x);
  gemcscpad_tree_->Branch("g_y", &gemcscpad_digi_.g_y);
  gemcscpad_tree_->Branch("g_z", &gemcscpad_digi_.g_z);
}

void GEMDigiAnalyzer::bookGEMCSCCoPadDigiTree()
{
  edm::Service<TFileService> fs;
  gemcsccopad_tree_ = fs->make<TTree>("GEMCSCCoPadDigiTree", "GEMCSCCoPadDigiTree");
  gemcsccopad_tree_->Branch("detId", &gemcsccopad_digi_.detId);
  gemcsccopad_tree_->Branch("region", &gemcsccopad_digi_.region);
  gemcsccopad_tree_->Branch("ring", &gemcsccopad_digi_.ring);
  gemcsccopad_tree_->Branch("station", &gemcsccopad_digi_.station);
  gemcsccopad_tree_->Branch("layer", &gemcsccopad_digi_.layer);
  gemcsccopad_tree_->Branch("chamber", &gemcsccopad_digi_.chamber);
  gemcsccopad_tree_->Branch("roll", &gemcsccopad_digi_.roll);
  gemcsccopad_tree_->Branch("pad", &gemcsccopad_digi_.pad);
  gemcsccopad_tree_->Branch("bx", &gemcsccopad_digi_.bx);
  gemcsccopad_tree_->Branch("x", &gemcsccopad_digi_.x);
  gemcsccopad_tree_->Branch("y", &gemcsccopad_digi_.y);
  gemcsccopad_tree_->Branch("g_r", &gemcsccopad_digi_.g_r);
  gemcsccopad_tree_->Branch("g_eta", &gemcsccopad_digi_.g_eta);
  gemcsccopad_tree_->Branch("g_phi", &gemcsccopad_digi_.g_phi);
  gemcsccopad_tree_->Branch("g_x", &gemcsccopad_digi_.g_x);
  gemcsccopad_tree_->Branch("g_y", &gemcsccopad_digi_.g_y);
  gemcsccopad_tree_->Branch("g_z", &gemcsccopad_digi_.g_z);
}

 void GEMDigiAnalyzer::bookSimTracksTree()
 {
   edm::Service< TFileService > fs;
   track_tree_ = fs->make<TTree>("TrackTree", "TrackTree");
   track_tree_->Branch("pt", &track_.pt);
   track_tree_->Branch("eta", &track_.eta);
   track_tree_->Branch("phi", &track_.phi);
   track_tree_->Branch("charge", &track_.charge);
   track_tree_->Branch("endcap", &track_.endcap);
   track_tree_->Branch("has_gem_sh", &track_.has_gem_sh);
   track_tree_->Branch("has_gem_sh2", &track_.has_gem_sh2);
   track_tree_->Branch("has_gem_dg", &track_.has_gem_dg);
   track_tree_->Branch("has_gem_dg2", &track_.has_gem_dg2);
   track_tree_->Branch("has_gem_pad", &track_.has_gem_pad);
   track_tree_->Branch("has_gem_pad2", &track_.has_gem_pad2);
   track_tree_->Branch("has_gem_copad", &track_.has_gem_copad);
   track_tree_->Branch("gem_sh_layer", &track_.gem_sh_layer);
   track_tree_->Branch("gem_dg_layer", &track_.gem_dg_layer);
   track_tree_->Branch("gem_pad_layer", &track_.gem_pad_layer);
   track_tree_->Branch("gem_sh_eta", &track_.gem_sh_eta);
   track_tree_->Branch("gem_dg_eta", &track_.gem_dg_eta);
   track_tree_->Branch("gem_pad_eta", &track_.gem_pad_eta);
   track_tree_->Branch("gem_sh_phi", &track_.gem_sh_phi);
   track_tree_->Branch("gem_dg_phi", &track_.gem_dg_phi);
   track_tree_->Branch("gem_pad_phi", &track_.gem_pad_phi);
 }

// ------------ method called for each event  ------------
void GEMDigiAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void GEMDigiAnalyzer::endJob() 
{
}

// ======= RPC ========
void GEMDigiAnalyzer::analyzeRPC()
{
  //Loop over RPC digi collection
  for(RPCDigiCollection::DigiRangeIterator cItr = rpc_digis->begin(); cItr != rpc_digis->end(); ++cItr)
  {
    RPCDetId id = (*cItr).first; 

    if (id.region() == 0) continue; // not interested in barrel

    const GeomDet* gdet = rpc_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const RPCRoll * roll = rpc_geo_->roll(id);

    rpc_digi_.detId = id();
    rpc_digi_.region = (Short_t) id.region();
    rpc_digi_.ring = (Short_t) id.ring();
    rpc_digi_.station = (Short_t) id.station();
    rpc_digi_.sector = (Short_t) id.sector();
    rpc_digi_.layer = (Short_t) id.layer();
    rpc_digi_.subsector = (Short_t) id.subsector();
    rpc_digi_.roll = (Short_t) id.roll();

    RPCDigiCollection::const_iterator digiItr; 
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      rpc_digi_.strip = (Short_t) digiItr->strip();
      rpc_digi_.bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());
      rpc_digi_.x = (Float_t) lp.x();
      rpc_digi_.y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      rpc_digi_.g_r = (Float_t) gp.perp();
      rpc_digi_.g_eta = (Float_t) gp.eta();
      rpc_digi_.g_phi = (Float_t) gp.phi();
      rpc_digi_.g_x = (Float_t) gp.x();
      rpc_digi_.g_y = (Float_t) gp.y();
      rpc_digi_.g_z = (Float_t) gp.z();

      // fill TTree
      rpc_tree_->Fill();
    }
  }
}

// ======= GEM ========
void GEMDigiAnalyzer::analyzeGEM()
{
  //Loop over GEM digi collection
  for(GEMDigiCollection::DigiRangeIterator cItr = gem_digis->begin(); cItr != gem_digis->end(); ++cItr)
  {
    GEMDetId id = (*cItr).first; 

    const GeomDet* gdet = gem_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = gem_geo_->etaPartition(id);

    gem_digi_.detId = id();
    gem_digi_.region = (Short_t) id.region();
    gem_digi_.ring = (Short_t) id.ring();
    gem_digi_.station = (Short_t) id.station();
    gem_digi_.layer = (Short_t) id.layer();
    gem_digi_.chamber = (Short_t) id.chamber();
    gem_digi_.roll = (Short_t) id.roll();

    GEMDigiCollection::const_iterator digiItr; 
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      gem_digi_.strip = (Short_t) digiItr->strip();
      gem_digi_.bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());
      gem_digi_.x = (Float_t) lp.x();
      gem_digi_.y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      gem_digi_.g_r = (Float_t) gp.perp();
      gem_digi_.g_eta = (Float_t) gp.eta();
      gem_digi_.g_phi = (Float_t) gp.phi();
      gem_digi_.g_x = (Float_t) gp.x();
      gem_digi_.g_y = (Float_t) gp.y();
      gem_digi_.g_z = (Float_t) gp.z();

      // fill TTree
      gem_tree_->Fill();
    }
  }
}


// ======= GEMCSCPad ========
void GEMDigiAnalyzer::analyzeGEMCSCPad()
{
  //Loop over GEMCSCPad digi collection
  for(GEMCSCPadDigiCollection::DigiRangeIterator cItr = gemcscpad_digis->begin(); cItr != gemcscpad_digis->end(); ++cItr)
    {
    GEMDetId id = (*cItr).first; 

    const GeomDet* gdet = gem_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = gem_geo_->etaPartition(id);

    gemcscpad_digi_.detId = id();
    gemcscpad_digi_.region = (Short_t) id.region();
    gemcscpad_digi_.ring = (Short_t) id.ring();
    gemcscpad_digi_.station = (Short_t) id.station();
    gemcscpad_digi_.layer = (Short_t) id.layer();
    gemcscpad_digi_.chamber = (Short_t) id.chamber();
    gemcscpad_digi_.roll = (Short_t) id.roll();

    GEMCSCPadDigiCollection::const_iterator digiItr; 
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      gemcscpad_digi_.pad = (Short_t) digiItr->pad();
      gemcscpad_digi_.bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfPad(digiItr->pad());
      gemcscpad_digi_.x = (Float_t) lp.x();
      gemcscpad_digi_.y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      gemcscpad_digi_.g_r = (Float_t) gp.perp();
      gemcscpad_digi_.g_eta = (Float_t) gp.eta();
      gemcscpad_digi_.g_phi = (Float_t) gp.phi();
      gemcscpad_digi_.g_x = (Float_t) gp.x();
      gemcscpad_digi_.g_y = (Float_t) gp.y();
      gemcscpad_digi_.g_z = (Float_t) gp.z();

      // fill TTree
      gemcscpad_tree_->Fill();
    }
  }
}


// ======= GEMCSCCoPad ========
void GEMDigiAnalyzer::analyzeGEMCSCCoPad()
{
  //Loop over GEMCSCPad digi collection
  for(GEMCSCPadDigiCollection::DigiRangeIterator cItr = gemcsccopad_digis->begin(); cItr != gemcsccopad_digis->end(); ++cItr)
  {
    GEMDetId id = (*cItr).first; 

    const GeomDet* gdet = gem_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = gem_geo_->etaPartition(id);

    gemcsccopad_digi_.detId = id();
    gemcsccopad_digi_.region = (Short_t) id.region();
    gemcsccopad_digi_.ring = (Short_t) id.ring();
    gemcsccopad_digi_.station = (Short_t) id.station();
    gemcsccopad_digi_.layer = (Short_t) id.layer();
    gemcsccopad_digi_.chamber = (Short_t) id.chamber();
    gemcsccopad_digi_.roll = (Short_t) id.roll();

    GEMCSCPadDigiCollection::const_iterator digiItr; 
    //loop over digis of given roll
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      gemcsccopad_digi_.pad = (Short_t) digiItr->pad();
      gemcsccopad_digi_.bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfPad(digiItr->pad());
      gemcsccopad_digi_.x = (Float_t) lp.x();
      gemcsccopad_digi_.y = (Float_t) lp.y();

      GlobalPoint gp = surface.toGlobal(lp);
      gemcsccopad_digi_.g_r = (Float_t) gp.perp();
      gemcsccopad_digi_.g_eta = (Float_t) gp.eta();
      gemcsccopad_digi_.g_phi = (Float_t) gp.phi();
      gemcsccopad_digi_.g_x = (Float_t) gp.x();
      gemcsccopad_digi_.g_y = (Float_t) gp.y();
      gemcsccopad_digi_.g_z = (Float_t) gp.z();

      // fill TTree
      gemcsccopad_tree_->Fill();
    }
  }
}

bool GEMDigiAnalyzer::isSimTrackGood(const SimTrack &t)
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

// ======= GEM Matching ========
void GEMDigiAnalyzer::analyzeTracks(edm::ParameterSet cfg_, const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const edm::SimVertexContainer & sim_vert = *sim_vertices.product();
  const edm::SimTrackContainer & sim_trks = *sim_tracks.product();

  for (auto& t: sim_trks)
  {
    if (!isSimTrackGood(t)) continue;
    
    // match hits and digis to this SimTrack
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, iEvent, iSetup);
    
    const SimHitMatcher&  match_sh = match.simhits();
    const GEMDigiMatcher& match_gd = match.gemDigis();
    //    const SimTrack &t = match_sh.trk();
    
    track_.pt = t.momentum().pt();
    track_.phi = t.momentum().phi();
    track_.eta = t.momentum().eta();
    track_.charge = t.charge();
    track_.endcap = (track_.eta > 0.) ? 1 : -1;
    track_.has_gem_sh = 0;
    track_.has_gem_sh2 = 0;
    track_.has_gem_dg = 0;
    track_.has_gem_dg2 = 0;
    track_.has_gem_pad = 0;
    track_.has_gem_pad2 = 0;
    track_.has_gem_copad = 0;
    
    // ** GEM SimHits ** //    
    auto gem_sh_ids_sch = match_sh.superChamberIdsGEM();
    for(auto d: gem_sh_ids_sch)
    {
      GEMDetId id(d);
      bool odd = id.chamber() %2 == 1;
      
      // at least one hit
      if (match_sh.hitsInSuperChamber(d).size() > 0)
      {
	if (odd) track_.has_gem_sh |= 1;
	else     track_.has_gem_sh |= 2;
      }
      // at least two hits
      if (match_sh.nLayersWithHitsInSuperChamber(d) > 1)
      {
	if (odd) track_.has_gem_sh2 |= 1;
	else     track_.has_gem_sh2 |= 2;
      }
    }
    
    auto gem_sh_ids_ch = match_sh.chamberIdsGEM();
    for(auto d: gem_sh_ids_ch)
    {
      GEMDetId id(d);
      track_.gem_sh_layer = id.layer();
      
      auto gem_simhits = match_sh.hitsInChamber(d);
      auto gem_sh_gp = match_sh.simHitsMeanPosition(gem_simhits);
      track_.gem_sh_eta = gem_sh_gp.eta();
      track_.gem_sh_phi = gem_sh_gp.phi();	      
    }

    // ** GEM Digis ** //    
    auto gem_dg_ids_sch = match_gd.superChamberIds();
    for(auto d: gem_dg_ids_sch)
    {
      GEMDetId id(d);
      bool odd = id.chamber() %2 == 1;
      
      // at least one digi
      if (match_gd.digisInSuperChamber(d).size() > 0)
      {
	if (odd) track_.has_gem_dg |= 1;
	else     track_.has_gem_dg |= 2;
      }
      // at least two digis
      if (match_gd.nLayersWithDigisInSuperChamber(d) > 1)
      {
	if (odd) track_.has_gem_dg2 |= 1;
	else     track_.has_gem_dg2 |= 2;
      }
      // at least one pad
      if (match_gd.padsInSuperChamber(d).size() > 0)
      {
	if (odd) track_.has_gem_pad |= 1;
	else     track_.has_gem_pad |= 2;
      }
      // at least two pad
      if (match_gd.nLayersWithPadsInSuperChamber(d) > 1)
      {
	if (odd) track_.has_gem_pad2 |= 1;
	else     track_.has_gem_pad2 |= 2;
      }
    }

    // ** GEMCSC CoPads ** //
    auto gem_dg_ids_sch_copad = match_gd.superChamberIdsWithCoPads();
    for(auto d: gem_dg_ids_sch_copad)
    {
      GEMDetId id(d);
      bool odd = id.chamber() %2 == 1;
      // has a copad
      if (match_gd.coPadsInSuperChamber(d).size() > 0)
      {
	if (odd) track_.has_gem_copad |= 1;
	else     track_.has_gem_copad |= 2;
      }
    }
    
    auto gem_dg_ids_ch = match_gd.chamberIds();
    for(auto d: gem_dg_ids_ch)
    {
      GEMDetId id(d);
      track_.gem_dg_layer = id.layer();
      
      auto gem_digis = match_gd.digisInChamber(d);
      auto gem_dg_gp = match_gd.digisMeanPosition(gem_digis);
      track_.gem_dg_eta = gem_dg_gp.eta();
      track_.gem_dg_phi = gem_dg_gp.phi();	      
      auto gem_pads = match_gd.padsInChamber(d);
      for (auto gem_pad: gem_pads)
      {
	auto gem_pad_gp = match_gd.digiPosition(gem_pad);	  
	track_.gem_pad_eta = gem_pad_gp.eta();
	track_.gem_pad_phi = gem_pad_gp.phi();	      
      }
    }      
    track_tree_->Fill();
  } // track loop
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GEMDigiAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMDigiAnalyzer);
