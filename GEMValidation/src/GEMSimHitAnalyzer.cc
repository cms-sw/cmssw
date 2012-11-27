// -*- C++ -*-
//
// Package:    GEMSimHitAnalyzer
// Class:      GEMSimHitAnalyzer
// 
/**\class GEMSimHitAnalyzer

 Description: Analyzer GEM SimHit information (as well as CSC & RPC SimHits). 
 To be used for GEM algorithm development.

 Notes:
 "DataFormats/MuonDetId/interface/GEMDetId.h" is not yet included in CMSSW. Do:
 cvs co -r V01-03-03 DataFormats/MuonDetId

 For GEM geometry use the packages given by the Vadim's twiki:
 https://twiki.cern.ch/twiki/bin/view/MPGD/GemSimulationsInstructionsCMSSW:
 cvs co -r V01-03-03      DataFormats/MuonDetId
 cvs co -r V01-09-07      Geometry/CMSCommonData
 cvs co -r V01-05-03      Geometry/ForwardCommonData
 cvs co -r V01-05-01      Geometry/HcalCommonData
 cvs co -r V01-07-07      Geometry/MuonCommonData
 cvs co -r V01-00-01      Geometry/MuonNumbering
 cvs co -r V01-03-01      Geometry/MuonSimData
 cvs co -r V02-08-09      SimG4CMS/Calo
 cvs co -r V01-01-06      SimG4CMS/Muon
 cvs co -r V01-04-07      SimG4CMS/ShowerLibraryProducer
 cvs co -r V05-16-05      SimG4Core/Application

 Then, you'll also need Marcello's additions:
 addpkg  Geometry/GEMGeometry V00-01-02 
 addpkg  Geometry/GEMGeometryBuilder  V00-01-02 
 addpkg  Geometry/Records V02-04-00 

 cvs co -r HEAD Geometry/CommonDetUnit 
 scp -r lxplus.cern.ch:/afs/cern.ch/user/m/mmaggi/public/UPGRADE/CommonDetUnit Geometry/ 

 cvs update -r V01-07-08 Geometry/MuonCommonData

 You'll need these before you can include, for example:
 #include "Geometry/GEMGeometry/interface/GEMGeometry.h"

*/
//
// Original Author:  Will Flanagan
//         Created:  Sat Nov 17 19:03:48 CST 2012
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH2.h"
#include "TTree.h"

using namespace edm;
using namespace std;


struct MyCSCSimHit
{  
   Int_t detUnitId, particleType;
   Float_t x, y, energyLoss, pabs, timeOfFlight;
   Int_t endcap, ring, station, chamber, layer;
   Float_t global_perp, global_eta, global_phi, global_x, global_y, global_z;
};


struct MyRPCSimHit
{  
   Int_t detUnitId, particleType;
   Float_t x, y, energyLoss, pabs, timeOfFlight;
   Int_t region, ring, station, sector, layer, subsector, roll;
   Float_t global_perp, global_eta, global_phi, global_x, global_y, global_z;
};


struct MyGEMSimHit
{  
   Int_t detUnitId, particleType;
   Float_t x, y, energyLoss, pabs, timeOfFlight;
   Int_t region, ring, station, layer, chamber, roll;
   Float_t global_perp, global_eta, global_phi, global_x, global_y, global_z;
};


class GEMSimHitAnalyzer : public edm::EDAnalyzer
{
   public:
      explicit GEMSimHitAnalyzer(const edm::ParameterSet&);
      ~GEMSimHitAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:

      void bookCSCSimHitsTrees();
      void bookRPCSimHitsTrees();
      void bookGEMSimHitsTrees();

      void analyzeCSC();
      void analyzeRPC();
      void analyzeGEM();

      // ----------member data ---------------------------

      TTree* csc_sh_tree;
      TTree* rpc_sh_tree;
      TTree* gem_sh_tree;

      TH2D * h_csc_localxy;
      TH2D * h_rpc_localxy;
      TH2D * h_gem_localxy;
      TH2D * h_csc_globalxy;
      TH2D * h_rpc_globalxy;
      TH2D * h_gem_globalxy;
      TH2D * h_csc_globalzx;
      TH2D * h_rpc_globalzx;
      TH2D * h_gem_globalzx;
      
      Handle<PSimHitContainer> CSCHits;
      Handle<PSimHitContainer> RPCHits;
      Handle<PSimHitContainer> GEMHits;
      
      ESHandle<CSCGeometry> csc_geom;
      ESHandle<RPCGeometry> rpc_geom;
      ESHandle<GEMGeometry> gem_geom;
      
      const CSCGeometry* csc_geometry;
      const RPCGeometry* rpc_geometry;
      const GEMGeometry* gem_geometry;
      
      PSimHitContainer::const_iterator itHit;
      
      MyCSCSimHit csc_sh;
      MyRPCSimHit rpc_sh;
      MyGEMSimHit gem_sh;
      
      GlobalPoint hitGP;

};

GEMSimHitAnalyzer::GEMSimHitAnalyzer(const edm::ParameterSet& iConfig)
{

   Service<TFileService> fs;

   h_csc_localxy = fs->make<TH2D>("h_csc_localxy","h_csc_localxy",50,25,25,50,25,25);
   h_rpc_localxy = fs->make<TH2D>("h_rpc_localxy","h_rpc_localxy",50,25,25,50,25,25);
   h_gem_localxy = fs->make<TH2D>("h_gem_localxy","h_gem_localxy",50,150,150,50,150,150);
   h_csc_globalxy = fs->make<TH2D>("h_csc_globalxy","h_csc_globalxy",50,-800,800,50,-800,800);
   h_rpc_globalxy = fs->make<TH2D>("h_rpc_globalxy","h_rpc_globalxy",50,-800,800,50,-800,800);
   h_gem_globalxy = fs->make<TH2D>("h_gem_globalxy","h_gem_globalxy",50,-800,800,50,-800,800);
   h_csc_globalzx = fs->make<TH2D>("h_csc_globalzx","h_csc_globalzx",50,-800,800,50,-800,800);
   h_rpc_globalzx = fs->make<TH2D>("h_rpc_globalzx","h_rpc_globalzx",50,-800,800,50,-800,800);
   h_gem_globalzx = fs->make<TH2D>("h_gem_globalzx","h_gem_globalzx",50,-800,800,50,-800,800);

   bookCSCSimHitsTrees();
   bookRPCSimHitsTrees();
   bookGEMSimHitsTrees();

}


GEMSimHitAnalyzer::~GEMSimHitAnalyzer()
{
}


//
// member functions
//

// ------------ method called for each event  ------------
void
GEMSimHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   iSetup.get<MuonGeometryRecord>().get(csc_geom);
   csc_geometry = &*csc_geom;

   iSetup.get<MuonGeometryRecord>().get(rpc_geom);
   rpc_geometry = &*rpc_geom;

   iSetup.get<MuonGeometryRecord>().get(gem_geom);
   gem_geometry = &*gem_geom;

   InputTag default_tag_csc("g4SimHits","MuonCSCHits");
   iEvent.getByLabel(default_tag_csc, CSCHits);
   if(CSCHits->size()) analyzeCSC();

   InputTag default_tag_rpc("g4SimHits","MuonRPCHits");
   iEvent.getByLabel(default_tag_rpc, RPCHits);
   if(RPCHits->size()) analyzeRPC();

   InputTag default_tag_gem("g4SimHits","MuonGEMHits");
   iEvent.getByLabel(default_tag_gem, GEMHits);
   if(GEMHits->size()) analyzeGEM();

}

void
GEMSimHitAnalyzer::bookCSCSimHitsTrees()
{

   Service<TFileService> fs;
   csc_sh_tree = fs->make<TTree>("CSCSimHitsTree2", "CSCSimHitsTree2");
   csc_sh_tree->Branch("csc_sh", &csc_sh,"detUnitId/I:particleType/I:x/F:y/F:energyLoss/F:pabs/F:timeOfFlight/F:endcap/I:ring/I:station/I:chamber/I:layer/I:global_perp/F:global_eta/F:global_phi/F:global_x/F:global_y/F:global_z/F");

}

void
GEMSimHitAnalyzer::bookRPCSimHitsTrees()
{

   Service<TFileService> fs;
   rpc_sh_tree = fs->make<TTree>("RPCSimHitsTree", "RPCSimHitsTree");
   rpc_sh_tree->Branch("rpc_sh", &rpc_sh,"detUnitId/I:particleType/I:x/F:y/F:energyLoss/F:pabs/F:timeOfFlight/F:region/I:ring/I:station/I:sector/I:layer/I:subsector/I:roll/I:global_perp/F:global_eta/F:global_phi/F:global_x/F:global_y/F:global_z/F");

}

void
GEMSimHitAnalyzer::bookGEMSimHitsTrees()
{  

   Service<TFileService> fs;
   gem_sh_tree = fs->make<TTree>("GEMSimHitsTree", "GEMSimHitsTree");
   gem_sh_tree->Branch("gem_sh",&gem_sh,"detUnitId/I:particleType/I:x/F:y/F:energyLoss/F:pabs/F:timeOfFlight/F:region/I:ring/I:station/I:layer/I:chamber/I:roll/I:global_perp/F:global_eta/F:global_phi/F:global_x/F:global_y/F:global_z/F");

}

void
GEMSimHitAnalyzer::analyzeCSC()
{

   for (itHit = CSCHits->begin(); itHit != CSCHits->end(); ++itHit) {
      
      CSCDetId id(itHit->detUnitId());

      if (id.station() != 1) continue; // here we care only about station 1
   
      csc_sh.detUnitId=itHit->detUnitId();
      csc_sh.particleType=itHit->particleType();
      csc_sh.x=itHit->localPosition().x();
      csc_sh.y=itHit->localPosition().y();
      csc_sh.energyLoss=itHit->energyLoss();
      csc_sh.pabs=itHit->pabs();
      csc_sh.timeOfFlight=itHit->timeOfFlight();
      
      h_csc_localxy->Fill(csc_sh.x,csc_sh.y);
      
      csc_sh.endcap=id.endcap();
      csc_sh.ring=id.ring();
      csc_sh.station=id.station();
      csc_sh.chamber=id.chamber();
      csc_sh.layer=id.layer();
      
      LocalPoint hitLP = itHit->localPosition();
      hitGP = csc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);

      csc_sh.global_perp=hitGP.perp();
      csc_sh.global_eta=hitGP.eta();
      csc_sh.global_phi=hitGP.phi();
      csc_sh.global_x=hitGP.x();
      csc_sh.global_y=hitGP.y();
      csc_sh.global_z=hitGP.z();

      h_csc_globalxy->Fill(csc_sh.global_x,csc_sh.global_y);
      h_csc_globalzx->Fill(csc_sh.global_z,csc_sh.global_x);

      csc_sh_tree->Fill();

   }

}

void
GEMSimHitAnalyzer::analyzeRPC()
{

   for (itHit = RPCHits->begin(); itHit != RPCHits->end(); ++itHit) {

      RPCDetId id(itHit->detUnitId());

      if (id.region() == 0) continue; // we don't care about barrel RPCs

      rpc_sh.detUnitId=itHit->detUnitId();
      rpc_sh.particleType=itHit->particleType();
      rpc_sh.x=itHit->localPosition().x();
      rpc_sh.y=itHit->localPosition().y();
      rpc_sh.energyLoss=itHit->energyLoss();
      rpc_sh.pabs=itHit->pabs();
      rpc_sh.timeOfFlight=itHit->timeOfFlight();
      
      h_rpc_localxy->Fill(rpc_sh.x,rpc_sh.y);
      
      rpc_sh.region=id.region();
      rpc_sh.ring=id.ring();
      rpc_sh.station=id.station();
      rpc_sh.sector=id.sector();
      rpc_sh.layer=id.layer();
      rpc_sh.subsector=id.subsector();
      rpc_sh.roll=id.roll();
      
      LocalPoint hitLP = itHit->localPosition();
      hitGP = rpc_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
      
      rpc_sh.global_perp=hitGP.perp();
      rpc_sh.global_eta=hitGP.eta();
      rpc_sh.global_phi=hitGP.phi();
      rpc_sh.global_x=hitGP.x();
      rpc_sh.global_y=hitGP.y();
      rpc_sh.global_z=hitGP.z();
      
      h_rpc_globalxy->Fill(rpc_sh.global_x,rpc_sh.global_y);
      h_rpc_globalzx->Fill(rpc_sh.global_z,rpc_sh.global_x);
      
      rpc_sh_tree->Fill();
   
   }

}

void
GEMSimHitAnalyzer::analyzeGEM()
{

   for (itHit = GEMHits->begin(); itHit != GEMHits->end(); ++itHit) {

      gem_sh.detUnitId=itHit->detUnitId();
      gem_sh.particleType=itHit->particleType();
      gem_sh.x=itHit->localPosition().x();
      gem_sh.y=itHit->localPosition().y();
      gem_sh.energyLoss=itHit->energyLoss();
      gem_sh.pabs=itHit->pabs();
      gem_sh.timeOfFlight=itHit->timeOfFlight();

      h_gem_localxy->Fill(gem_sh.x,gem_sh.y);

      GEMDetId id(itHit->detUnitId());

      gem_sh.region=id.region();
      gem_sh.ring=id.ring();
      gem_sh.station=id.station();
      gem_sh.layer=id.layer();
      gem_sh.chamber=id.chamber();
      gem_sh.roll=id.roll();
      
      LocalPoint hitLP = itHit->localPosition();
      hitGP = gem_geometry->idToDet(itHit->detUnitId())->surface().toGlobal(hitLP);
      
      gem_sh.global_perp=hitGP.perp();
      gem_sh.global_eta=hitGP.eta();
      gem_sh.global_phi=hitGP.phi();
      gem_sh.global_x=hitGP.x();
      gem_sh.global_y=hitGP.y();
      gem_sh.global_z=hitGP.z();
      
      h_gem_globalxy->Fill(gem_sh.global_x,gem_sh.global_y);
      h_gem_globalzx->Fill(gem_sh.global_z,gem_sh.global_x);
      
      gem_sh_tree->Fill();
   
   }

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
GEMSimHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMSimHitAnalyzer);

