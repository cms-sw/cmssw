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
// $Id$
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
#include "TH1.h"
#include "TTree.h"
#include "TFile.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"


///Data Format
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
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

using namespace std;
using namespace edm;

//
// class declaration
//



struct MyRPCDigi
{  
   Int_t detId;
   Short_t region, ring, station, sector, layer, subsector, roll;
   Short_t strip, bx;
   Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};

struct MyGEMDigi
{  
   Int_t detId;
   Short_t region, ring, station, layer, chamber, roll;
   Short_t strip, bx;
   Float_t g_r, g_eta, g_phi, g_x, g_y, g_z;
};


class GEMDigiAnalyzer : public edm::EDAnalyzer 
{
public:

  explicit GEMDigiAnalyzer(const edm::ParameterSet&);
  ~GEMDigiAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  // ----------member data ---------------------------


  // --- configuration parameters ---
  
  int verbosity_;
  
  edm::ESHandle<RPCGeometry> rpc_geo_;
  //edm::ESHandle<CSCGeometry> csc_geo_;
  edm::ESHandle<GEMGeometry> gem_geo_;

  edm::InputTag input_tag_rpc_;
  //edm::InputTag input_tag_csc_;
  edm::InputTag input_tag_gem_;
  

  // --- histograms and digis ---
  
  TH1D *h_rpc_strip_;
  TH1D *h_gem_strip_;

  MyRPCDigi rpc_digi_;
  //MyCSCDigi csc_digi_;
  MyGEMDigi gem_digi_;
  
  TTree* rpc_tree_;
  //TTree* csc_tree_;
  TTree* gem_tree_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GEMDigiAnalyzer::GEMDigiAnalyzer(const edm::ParameterSet&iConfig)
{
  //now do what ever initialization is needed
  
  verbosity_ = iConfig.getUntrackedParameter<int>("verbosity", 0);

  input_tag_rpc_ = iConfig.getUntrackedParameter<InputTag>("inputTagRPC", InputTag("simMuonRPCDigis"));
  //input_tag_csc_ = iConfig.getUntrackedParameter<InputTag>("inputTagCSC", InputTag("simMuonICSCDigis"));
  input_tag_gem_ = iConfig.getUntrackedParameter<InputTag>("inputTagGEM", InputTag("simMuonGEMDigis"));
  
  edm::Service<TFileService> fs;

  h_rpc_strip_ = fs->make<TH1D>("rpc_digi_strip", "rpc_digi_strip", 100, 0, 100);

  rpc_tree_ = fs->make<TTree>("RPCDigiTree", "RPCDigiTree");
  //rpc_tree_->Branch("digi", &rpc_digi_, "detId/I:region/S:ring:station:sector:layer:subsector:roll:strip:bx:g_r/F:g_eta:g_phi:g_x:g_y:g_z");
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
  rpc_tree_->Branch("g_r", &rpc_digi_.g_r);
  rpc_tree_->Branch("g_eta", &rpc_digi_.g_eta);
  rpc_tree_->Branch("g_phi", &rpc_digi_.g_phi);
  rpc_tree_->Branch("g_x", &rpc_digi_.g_x);
  rpc_tree_->Branch("g_y", &rpc_digi_.g_y);
  rpc_tree_->Branch("g_z", &rpc_digi_.g_z);
  

  h_gem_strip_ = fs->make<TH1D>("gem_digi_strip", "gem_digi_strip", 400, 0, 400);

  gem_tree_ = fs->make<TTree>("GEMDigiTree", "GEMDigiTree");
  //gem_tree_->Branch("digi", &gem_digi_, "detId/I:region/S:ring:station:layer:chamber:roll:strip:bx:g_r/F:g_eta:g_phi:g_x:g_y:g_z");
  gem_tree_->Branch("detId", &gem_digi_.detId);
  gem_tree_->Branch("region", &gem_digi_.region);
  gem_tree_->Branch("ring", &gem_digi_.ring);
  gem_tree_->Branch("station", &gem_digi_.station);
  gem_tree_->Branch("layer", &gem_digi_.layer);
  gem_tree_->Branch("chamber", &gem_digi_.chamber);
  gem_tree_->Branch("roll", &gem_digi_.roll);
  gem_tree_->Branch("strip", &gem_digi_.strip);
  gem_tree_->Branch("bx", &gem_digi_.bx);
  gem_tree_->Branch("g_r", &gem_digi_.g_r);
  gem_tree_->Branch("g_eta", &gem_digi_.g_eta);
  gem_tree_->Branch("g_phi", &gem_digi_.g_phi);
  gem_tree_->Branch("g_x", &gem_digi_.g_x);
  gem_tree_->Branch("g_y", &gem_digi_.g_y);
  gem_tree_->Branch("g_z", &gem_digi_.g_z);
}


GEMDigiAnalyzer::~GEMDigiAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  //delete file_;

}


//
// member functions
//

// ------------ method called for each event  ------------

void GEMDigiAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void GEMDigiAnalyzer::endJob() 
{
}


void GEMDigiAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // ======= RPC ========
  
  edm::Handle<RPCDigiCollection> rpc_digis;
  iEvent.getByLabel(input_tag_rpc_, rpc_digis);

  //Loop over RPC digi collection
  for(RPCDigiCollection::DigiRangeIterator cItr = rpc_digis->begin(); cItr != rpc_digis->end(); ++cItr)
  {
    RPCDetId id = (*cItr).first; 

    if (id.region() == 0) continue; // not interested in barrel

    const GeomDet* gdet = rpc_geo_->idToDet(id);
    const BoundPlane & surface = gdet->surface();

    const RPCRoll * roll = rpc_geo_->roll(id);

    /*
    // get roll name
    RPCGeomServ RPCname(id);
    string nameRoll = RPCname.name();
    stringstream os;
   
    // get info
    int region = id.region();
    int ring;
    string ringType;
    string regionType;
    ringType =  "Disk";
    regionType = "Endcap";
    ring = region * id.station();
    
    int station = id.station();
    int sector = id.sector();
    cout<<" "<<endl;
    cout<<" RPC DetId: "<<setw(12)<<id()<<" a.k.a. "<<setw(18)<<nameRoll<<" which is in "<<setw(6)<<regionType<<" "<<setw(5)<<ringType<<" "<<setw(2)<<ring;
    cout<<" station "<<setw(2)<<station<<" sector "<<setw(2)<<sector<<endl;
    cout<<" ---------------------------------------------------------------------------------------------"<<endl;
    */
    
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
      GlobalPoint gp = surface.toGlobal(lp);
      rpc_digi_.g_r = (Float_t) gp.perp();
      rpc_digi_.g_eta = (Float_t) gp.eta();
      rpc_digi_.g_phi = (Float_t) gp.phi();
      rpc_digi_.g_x = (Float_t) gp.x();
      rpc_digi_.g_y = (Float_t) gp.y();
      rpc_digi_.g_z = (Float_t) gp.z();

      if (verbosity_ > 0) cout<<"  RPCDigi: strip = "<<setw(2)<<rpc_digi_.strip<<" bx = "<<setw(2)<<rpc_digi_.bx<<" "<<lp<<" "<<gp<<" "<<rpc_digi_.g_r<<endl;

      // Fill histograms
      h_rpc_strip_->Fill(digiItr->strip());

      // fill TTree
      rpc_tree_->Fill();
    }
  }


  // ======= GEM ========
  
  edm::Handle<GEMDigiCollection> gem_digis;
  iEvent.getByLabel(input_tag_gem_, gem_digis);

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
      GlobalPoint gp = surface.toGlobal(lp);
      gem_digi_.g_r = (Float_t) gp.perp();
      gem_digi_.g_eta = (Float_t) gp.eta();
      gem_digi_.g_phi = (Float_t) gp.phi();
      gem_digi_.g_x = (Float_t) gp.x();
      gem_digi_.g_y = (Float_t) gp.y();
      gem_digi_.g_z = (Float_t) gp.z();

      if (verbosity_ > 0) cout<<"  GEMDigi: strip = "<<setw(2)<<gem_digi_.strip<<" bx = "<<setw(2)<<gem_digi_.bx<<" "<<lp<<" "<<gp<<" "<<gem_digi_.g_r<<endl;

      // Fill histograms
      h_gem_strip_->Fill(digiItr->strip());

      // fill TTree
      gem_tree_->Fill();
    }
    if (verbosity_ > 0) cout<<" ---------------------------------------------------------------------------------------------"<<endl;
  }
 
}


// ------------ method called when starting to processes a run  ------------
void GEMDigiAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(rpc_geo_);
  iSetup.get<MuonGeometryRecord>().get(gem_geo_);
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
