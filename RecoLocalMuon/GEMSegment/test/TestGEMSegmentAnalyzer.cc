// -*- C++ -*-
//
// Package:    TestGEMSegmentAnalyzer
// Class:      TestGEMSegmentAnalyzer
// 
/**\class TestGEMSegmentAnalyzer TestGEMSegmentAnalyzer.cc MyAnalyzers/TestGEMSegmentAnalyzer/src/TestGEMSegmentAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marcello Maggi 

// system include files
#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>


// root include files
#include "TFile.h"
#include "TH1F.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMRecHit/interface/GEMSegmentCollection.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Math/interface/deltaPhi.h"

//
// class declaration
//

class TestGEMSegmentAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestGEMSegmentAnalyzer(const edm::ParameterSet&);
      ~TestGEMSegmentAnalyzer();



   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
  edm::ESHandle<GEMGeometry> gemGeom;
  edm::ESHandle<CSCGeometry> cscGeom;

  edm::EDGetTokenT<reco::GenParticleCollection> GENParticle_Token;
  edm::EDGetTokenT<edm::HepMCProduct>           HEPMCCol_Token;
  edm::EDGetTokenT<edm::SimTrackContainer>      SIMTrack_Token;
  edm::EDGetTokenT<CSCSegmentCollection>        CSCSegment_Token;
  edm::EDGetTokenT<GEMSegmentCollection>        GEMSegment_Token;

  std::unique_ptr<TH1F> GEN_eta, GEN_phi, SIM_eta, SIM_phi;
  std::unique_ptr<TH1F> GE11_eta, GE11_phi, GE21_eta, GE21_phi;
  std::unique_ptr<TH1F> GE11_Delta_eta, GE11_Delta_phi;
  std::unique_ptr<TH1F> GE21_Delta_eta, GE21_Delta_phi;

  std::string rootFileName;
  std::unique_ptr<TFile> outputfile;
  std::unique_ptr<TH1F> GE11_numhits;
  std::unique_ptr<TH1F> GE11_fitchi2;
  std::unique_ptr<TH1F> GE11_fitndof;
  std::unique_ptr<TH1F> GE11_fitchi2ndof;
  std::unique_ptr<TH1F> GE11_Residuals_x;
  std::unique_ptr<TH1F> GE11_Residuals_l1_x;
  std::unique_ptr<TH1F> GE11_Residuals_l2_x;
  std::unique_ptr<TH1F> GE11_Pull_x;
  std::unique_ptr<TH1F> GE11_Pull_l1_x;
  std::unique_ptr<TH1F> GE11_Pull_l2_x;
  std::unique_ptr<TH1F> GE11_Residuals_y;
  std::unique_ptr<TH1F> GE11_Residuals_l1_y;
  std::unique_ptr<TH1F> GE11_Residuals_l2_y;
  std::unique_ptr<TH1F> GE11_Pull_y;
  std::unique_ptr<TH1F> GE11_Pull_l1_y;
  std::unique_ptr<TH1F> GE11_Pull_l2_y;

  std::unique_ptr<TH1F> GE21_numhits;
  std::unique_ptr<TH1F> GE21_fitchi2;
  std::unique_ptr<TH1F> GE21_fitndof;
  std::unique_ptr<TH1F> GE21_fitchi2ndof;
  std::unique_ptr<TH1F> GE21_Residuals_x;
  std::unique_ptr<TH1F> GE21_Residuals_l1_x;
  std::unique_ptr<TH1F> GE21_Residuals_l2_x;
  std::unique_ptr<TH1F> GE21_Residuals_l3_x;
  std::unique_ptr<TH1F> GE21_Residuals_l4_x;
  std::unique_ptr<TH1F> GE21_Pull_x;
  std::unique_ptr<TH1F> GE21_Pull_l1_x;
  std::unique_ptr<TH1F> GE21_Pull_l2_x;
  std::unique_ptr<TH1F> GE21_Pull_l3_x;
  std::unique_ptr<TH1F> GE21_Pull_l4_x;
  std::unique_ptr<TH1F> GE21_Residuals_y;
  std::unique_ptr<TH1F> GE21_Residuals_l1_y;
  std::unique_ptr<TH1F> GE21_Residuals_l2_y;
  std::unique_ptr<TH1F> GE21_Residuals_l3_y;
  std::unique_ptr<TH1F> GE21_Residuals_l4_y;
  std::unique_ptr<TH1F> GE21_Pull_y;
  std::unique_ptr<TH1F> GE21_Pull_l1_y;
  std::unique_ptr<TH1F> GE21_Pull_l2_y;
  std::unique_ptr<TH1F> GE21_Pull_l3_y;
  std::unique_ptr<TH1F> GE21_Pull_l4_y;

};

//
// constants, enums and typedefs
//
// constructors and destructor
//
TestGEMSegmentAnalyzer::TestGEMSegmentAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  GENParticle_Token = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));
  HEPMCCol_Token    = consumes<edm::HepMCProduct>(edm::InputTag("generator"));
  SIMTrack_Token    = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  CSCSegment_Token  = consumes<CSCSegmentCollection>(edm::InputTag("cscSegments"));
  GEMSegment_Token  = consumes<GEMSegmentCollection>(edm::InputTag("gemSegments"));

  rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");
  outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));

  GEN_eta = std::unique_ptr<TH1F>(new TH1F("GEN_eta","GEN_eta",100,-2.50,2.50));
  GEN_phi = std::unique_ptr<TH1F>(new TH1F("GEN_phi","GEN_phi",144,-3.14,3.14));
  SIM_eta = std::unique_ptr<TH1F>(new TH1F("SIM_eta","SIM_eta",100,-2.50,2.50));
  SIM_phi = std::unique_ptr<TH1F>(new TH1F("SIM_phi","SIM_phi",144,-3.14,3.14));

  GE11_eta = std::unique_ptr<TH1F>(new TH1F("GE11_eta","GE11_eta",100,-2.50,2.50));
  GE11_phi = std::unique_ptr<TH1F>(new TH1F("GE11_phi","GE11_phi",144,-3.14,3.14));
  GE21_eta = std::unique_ptr<TH1F>(new TH1F("GE21_eta","GE21_eta",100,-2.50,2.50));
  GE21_phi = std::unique_ptr<TH1F>(new TH1F("GE21_phi","GE21_phi",144,-3.14,3.14));

  GE11_Delta_eta = std::unique_ptr<TH1F>(new TH1F("GE11_Delta_eta","GE11_Delta_eta",100,-0.5,0.5));
  GE11_Delta_phi = std::unique_ptr<TH1F>(new TH1F("GE11_Delta_phi","GE11_Delta_phi",100,-0.5,0.5));
  GE21_Delta_eta = std::unique_ptr<TH1F>(new TH1F("GE21_Delta_eta","GE21_Delta_eta",100,-0.5,0.5));
  GE21_Delta_phi = std::unique_ptr<TH1F>(new TH1F("GE21_Delta_phi","GE21_Delta_phi",100,-0.5,0.5));

  GE11_fitchi2 = std::unique_ptr<TH1F>(new TH1F("GE11_chi2","GE11_chi2",11,-0.5,10.5)); 
  GE11_fitndof = std::unique_ptr<TH1F>(new TH1F("GE11_ndf","GE11_ndf",11,-0.5,10.5)); 
  GE11_fitchi2ndof = std::unique_ptr<TH1F>(new TH1F("GE11_chi2Vsndf","GE11_chi2Vsndf",50,0.,5.)); 
  GE11_numhits = std::unique_ptr<TH1F>(new TH1F("GE11_NumberOfHits","GE11_NumberOfHits",11,-0.5,10.5)); 
  GE11_Residuals_x    = std::unique_ptr<TH1F>(new TH1F("xGE11Res","xGE11Res",100,-0.5,0.5));
  GE11_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE11Res_l1","xGE11Res_l1",100,-0.5,0.5));
  GE11_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE11Res_l2","xGE11Res_l2",100,-0.5,0.5));
  GE11_Pull_x    = std::unique_ptr<TH1F>(new TH1F("xGE11Pull","xGE11Pull",100,-5.,5.));
  GE11_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE11Pull_l1","xGE11Pull_l1",100,-5.,5.));
  GE11_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE11Pull_l2","xGE11Pull_l2",100,-5.,5.));
  GE11_Residuals_y    = std::unique_ptr<TH1F>(new TH1F("yGE11Res","yGE11Res",100,-5.,5.));
  GE11_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE11Res_l1","yGE11Res_l1",100,-5.,5.));
  GE11_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE11Res_l2","yGE11Res_l2",100,-5.,5.));
  GE11_Pull_y    = std::unique_ptr<TH1F>(new TH1F("yGE11Pull","yGE11Pull",100,-5.,5.));
  GE11_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE11Pull_l1","yGE11Pull_l1",100,-5.,5.));
  GE11_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE11Pull_l2","yGE11Pull_l2",100,-5.,5.));

  GE21_fitchi2 = std::unique_ptr<TH1F>(new TH1F("GE21_chi2","GE21_chi2",11,0.5,10.5)); 
  GE21_fitndof = std::unique_ptr<TH1F>(new TH1F("GE21_ndf","GE21_ndf",11,-0.5,10.5)); 
  GE21_fitchi2ndof = std::unique_ptr<TH1F>(new TH1F("GE21_chi2Vsndf","GE21_chi2Vsndf",50,0.,5.)); 
  GE21_numhits = std::unique_ptr<TH1F>(new TH1F("GE21_NumberOfHits","GE21_NumberOfHits",11,-0.5,10.5)); 
  GE21_Residuals_x    = std::unique_ptr<TH1F>(new TH1F("xGE21Res","xGE21Res",100,-0.5,0.5));
  GE21_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21Res_l1","xGE21Res_l1",100,-0.5,0.5));
  GE21_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21Res_l2","xGE21Res_l2",100,-0.5,0.5));
  GE21_Residuals_l3_x = std::unique_ptr<TH1F>(new TH1F("xGE21Res_l3","xGE21Res_l3",100,-0.5,0.5));
  GE21_Residuals_l4_x = std::unique_ptr<TH1F>(new TH1F("xGE21Res_l4","xGE21Res_l4",100,-0.5,0.5));
  GE21_Pull_x    = std::unique_ptr<TH1F>(new TH1F("xGE21Pull","xGE21Pull",100,-5.,5.));
  GE21_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xGE21Pull_l1","xGE21Pull_l1",100,-5.,5.));
  GE21_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xGE21Pull_l2","xGE21Pull_l2",100,-5.,5.));
  GE21_Pull_l3_x = std::unique_ptr<TH1F>(new TH1F("xGE21Pull_l3","xGE21Pull_l3",100,-5.,5.));
  GE21_Pull_l4_x = std::unique_ptr<TH1F>(new TH1F("xGE21Pull_l4","xGE21Pull_l4",100,-5.,5.));
  GE21_Residuals_y    = std::unique_ptr<TH1F>(new TH1F("yGE21Res","yGE21Res",100,-5.,5.));
  GE21_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21Res_l1","yGE21Res_l1",100,-5.,5.));
  GE21_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21Res_l2","yGE21Res_l2",100,-5.,5.));
  GE21_Residuals_l3_y = std::unique_ptr<TH1F>(new TH1F("yGE21Res_l3","yGE21Res_l3",100,-5.,5.));
  GE21_Residuals_l4_y = std::unique_ptr<TH1F>(new TH1F("yGE21Res_l4","yGE21Res_l4",100,-5.,5.));
  GE21_Pull_y    = std::unique_ptr<TH1F>(new TH1F("yGE21Pull","yGE21Pull",100,-5.,5.));
  GE21_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yGE21Pull_l1","yGE21Pull_l1",100,-5.,5.));
  GE21_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yGE21Pull_l2","yGE21Pull_l2",100,-5.,5.));
  GE21_Pull_l3_y = std::unique_ptr<TH1F>(new TH1F("yGE21Pull_l3","yGE21Pull_l3",100,-5.,5.));
  GE21_Pull_l4_y = std::unique_ptr<TH1F>(new TH1F("yGE21Pull_l4","yGE21Pull_l4",100,-5.,5.));
}


TestGEMSegmentAnalyzer::~TestGEMSegmentAnalyzer()
{

  GEN_eta->Write();
  GEN_phi->Write();
  SIM_eta->Write();
  SIM_phi->Write();

  GE11_eta->Write();
  GE11_phi->Write();
  GE21_eta->Write();
  GE21_phi->Write();
  GE11_Delta_eta->Write();
  GE11_Delta_phi->Write();
  GE21_Delta_eta->Write();
  GE21_Delta_phi->Write();

  GE11_fitchi2->Write();
  GE11_fitndof->Write();
  GE11_fitchi2ndof->Write();
  GE11_numhits->Write();
  GE11_Residuals_x->Write();
  GE11_Residuals_l1_x->Write();
  GE11_Residuals_l2_x->Write();
  GE11_Pull_x->Write();
  GE11_Pull_l1_x->Write();
  GE11_Pull_l2_x->Write();
  GE11_Residuals_y->Write();
  GE11_Residuals_l1_y->Write();
  GE11_Residuals_l2_y->Write();
  GE11_Pull_y->Write();
  GE11_Pull_l1_y->Write();
  GE11_Pull_l2_y->Write();

  GE21_fitchi2->Write();
  GE21_fitndof->Write();
  GE21_fitchi2ndof->Write();
  GE21_numhits->Write();
  GE21_Residuals_x->Write();
  GE21_Residuals_l1_x->Write();
  GE21_Residuals_l2_x->Write();
  GE21_Residuals_l3_x->Write();
  GE21_Residuals_l4_x->Write();
  GE21_Pull_x->Write();
  GE21_Pull_l1_x->Write();
  GE21_Pull_l2_x->Write();
  GE21_Pull_l3_x->Write();
  GE21_Pull_l4_x->Write();
  GE21_Residuals_y->Write();
  GE21_Residuals_l1_y->Write();
  GE21_Residuals_l2_y->Write();
  GE21_Residuals_l3_y->Write();
  GE21_Residuals_l4_y->Write();
  GE21_Pull_y->Write();
  GE21_Pull_l1_y->Write();
  GE21_Pull_l2_y->Write();
  GE21_Pull_l3_y->Write();
  GE21_Pull_l4_y->Write();
}


//
// member functions
//

// ------------ method called for each event  ------------
void
TestGEMSegmentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  iSetup.get<MuonGeometryRecord>().get(gemGeom);
  iSetup.get<MuonGeometryRecord>().get(cscGeom);

  // ================
  // GEM Segments
  // ================
  edm::Handle<GEMSegmentCollection> gemSegmentCollection;
  iEvent.getByToken(GEMSegment_Token, gemSegmentCollection);

  std::cout <<"Number of GEM Segments in this event: "<<gemSegmentCollection->size()<<"\n"<<std::endl;

  // Loop over GEM Segments
  // ======================
  for (auto gems = gemSegmentCollection->begin(); gems != gemSegmentCollection->end(); ++gems) {

    std::cout<< "   Analyzing GEM Segment: \n   ---------------------- \n   "<<(*gems)<<std::endl;

    // obtain GEM DetId from GEMSegment ==> GEM Chamber 
    // and obtain corresponding GEMChamber from GEM Geometry
    // (GE1/1 --> station 1; GE2/1 --> station 3)
    GEMDetId id = gems->gemDetId();
    auto chamb = gemGeom->chamber(id); 

    // calculate Local & Global Position & Direction of GEM Segment
    auto segLP = gems->localPosition();
    auto segLD = gems->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);


    // obtain constituting GEM rechits
    auto gemrhs = gems->specificRecHits();

    // cout
    std::cout <<"   "<<std::endl;
    std::cout <<"   GEM Segment DetID "<<id<<" = "<<id.rawId()<<std::endl;
    std::cout <<"   Locl Position "<< segLP <<" Locl Direction "<<segLD<<std::endl;
    std::cout <<"   Glob Position "<< segGP <<" Glob Direction "<<segGD<<std::endl;
    std::cout <<"   Glob Pos  eta "<<segGP.eta()  << " Glob Pos phi " <<segGP.phi()<<std::endl;
    std::cout <<"   Locl Dir thet "<<segLD.theta()<< " Locl Dir phi " <<segLD.phi()<<std::endl;
    std::cout <<"   Glob Dir thet "<<segGD.theta()<< " Glob Dir phi " <<segGD.phi()<<std::endl;
    std::cout <<"   Chi2 = "<<gems->chi2()<<" ndof = "<<gems->degreesOfFreedom()<<" ==> chi2/ndof = "<<gems->chi2()*1.0/gems->degreesOfFreedom()<<std::endl;
    std::cout <<"   Number of RecHits "<<gemrhs.size()<<std::endl;
    std::cout <<"   "<<std::endl;


    // loop on rechits ... 
    // ===================
    // take layer local position -> global -> ensemble local position same frame as segment
    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){

      // GEM RecHit DetId & EtaPartition
      auto gemid = rh->gemId();
      auto rhr = gemGeom->etaPartition(gemid);

      // GEM RecHit Local & Global Position
      auto rhLP = rh->localPosition();
      auto erhLEP = rh->localPositionError();
      auto rhGP = rhr->toGlobal(rhLP); 

      std::cout <<"      const GEMRecHit in DetId "<<gemid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<rhGP.eta()<<" phi = "<<rhGP.phi()<<std::endl;


      // GEM RecHit Local Position in GEM Segment Chamber Frame
      auto rhLPSegm = chamb->toLocal(rhGP);

      // GEM Segment extrapolated to Layer of GEM RecHit in GEM Segment Local Frame
      float xe  = segLP.x()+segLD.x()*rhLPSegm.z()/segLD.z();
      float ye  = segLP.y()+segLD.y()*rhLPSegm.z()/segLD.z();
      float ze = rhLPSegm.z();
      LocalPoint extrPoint(xe,ye,ze);                          // in segment rest frame
      auto extSegm = rhr->toLocal(chamb->toGlobal(extrPoint)); // in chamber restframe
            std::cout <<"      GEM Layer Id "<<rh->gemId()<<"  error on the local point "<<  erhLEP
		<<"\n-> Ensemble Rest Frame  RH local  position "<<rhLPSegm<<"  Segment extrapolation "<<extrPoint
		<<"\n-> Layer Rest Frame  RH local  position "<<rhLP<<"  Segment extrapolation "<<extSegm
		<<std::endl;
      

      if(gemid.station()==1) {
	GE11_fitchi2->Fill(gems->chi2());
	GE11_fitndof->Fill(gems->degreesOfFreedom());
	GE11_fitchi2ndof->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
	GE11_numhits->Fill(gems->nRecHits());
	GE11_Residuals_x->Fill(rhLP.x()-extSegm.x());
	GE11_Residuals_y->Fill(rhLP.y()-extSegm.y());
	GE11_Pull_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	GE11_Pull_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	switch (gemid.layer()){
	case 1:
	  GE11_Residuals_l1_x->Fill(rhLP.x()-extSegm.x());
	  GE11_Residuals_l1_y->Fill(rhLP.y()-extSegm.y());
	  GE11_Pull_l1_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	  GE11_Pull_l1_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	  break;
	case 2:
	  GE11_Residuals_l2_x->Fill(rhLP.x()-extSegm.x());
	  GE11_Residuals_l2_y->Fill(rhLP.y()-extSegm.y());
	  GE11_Pull_l2_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	  GE11_Pull_l2_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	  break;
	  std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	}
      }
      else if(gemid.station()==2 || gemid.station()==3) {
	GE21_fitchi2->Fill(gems->chi2());
	GE21_fitndof->Fill(gems->degreesOfFreedom());
	GE21_fitchi2ndof->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
        GE21_numhits->Fill(gems->nRecHits());
	GE21_Residuals_x->Fill(rhLP.x()-extSegm.x());
	GE21_Residuals_y->Fill(rhLP.y()-extSegm.y());
	GE21_Pull_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	GE21_Pull_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	if(gemid.station()==2) {
	  switch (gemid.layer()){
	  case 1:
	    GE21_Residuals_l1_x->Fill(rhLP.x()-extSegm.x());
	    GE21_Residuals_l1_y->Fill(rhLP.y()-extSegm.y());
	    GE21_Pull_l1_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l1_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	    break;
	  case 2:
	    GE21_Residuals_l2_x->Fill(rhLP.x()-extSegm.x());
	    GE21_Residuals_l2_y->Fill(rhLP.y()-extSegm.y());
	    GE21_Pull_l2_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l2_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	    break;
	  default:
	  std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	  }
	}
	else if (gemid.station()==3) {
	  switch (gemid.layer()) {
	  case 1:
	    GE21_Residuals_l3_x->Fill(rhLP.x()-extSegm.x());
	    GE21_Residuals_l3_y->Fill(rhLP.y()-extSegm.y());
	    GE21_Pull_l3_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l3_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	    break;
	  case 2:
	    GE21_Residuals_l4_x->Fill(rhLP.x()-extSegm.x());
	    GE21_Residuals_l4_y->Fill(rhLP.y()-extSegm.y());
	    GE21_Pull_l4_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	    GE21_Pull_l4_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	    break;
	  default:
	    std::cout <<"      Unphysical GEM layer "<<gemid<<std::endl;
	  }
	}
      }
      else {}

    }
    std::cout<<"\n"<<std::endl;
  }
  std::cout<<"\n"<<std::endl;


  // Lets try to follow the muon trajectory
  // Piece of code optimized for single muon gun
  // Check first negative endcap, then positive endcap
  // Print position & direction of GEM & CSC segments
  // GE1/1 - ME1/1 - GE2/1 - ME2/1

  // Handles
  edm::Handle<reco::GenParticleCollection>      genParticles;
  iEvent.getByToken(GENParticle_Token, genParticles);

  edm::Handle<edm::HepMCProduct> hepmcevent;
  iEvent.getByToken(HEPMCCol_Token, hepmcevent);

  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByToken(SIMTrack_Token,SimTk);

  edm::Handle<CSCSegmentCollection> cscSegmentCollection;
  iEvent.getByToken(CSCSegment_Token, cscSegmentCollection);

  /*
  edm::Handle<reco::GenParticleCollection>      genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  std::vector<SimTrack> theSimTracks;
  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByLabel("g4SimHits",SimTk);
  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  edm::Handle<CSCSegmentCollection> cscSegmentCollection;
  iEvent.getByLabel("cscSegments", cscSegmentCollection);

  std::vector<SimTrack> theSimTracks;
  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  */

  // Containers
  // std::vector< std::unique_ptr<reco::GenParticle> > GEN_muons_pos, GEN_muons_neg;
  std::vector< std::unique_ptr<HepMC::GenParticle> >   GEN_muons_pos, GEN_muons_neg;
  std::vector< std::unique_ptr<SimTrack> >             SIM_muons_pos, SIM_muons_neg;   
  std::vector< std::unique_ptr<GEMSegment> >           GE11_segs_pos, GE11_segs_neg, GE21_segs_pos, GE21_segs_neg;
  std::vector< std::unique_ptr<CSCSegment> >           ME11_segs_pos, ME11_segs_neg, ME21_segs_pos, ME21_segs_neg;

  // Gen & Sim Muon
  // for(unsigned int i=0; i<genParticles->size(); ++i) {
  // std::unique_ptr<reco::GenParticle> g = &((*genParticles)[i]);
  /*
  for(reco::GenParticleCollection::const_iterator it=genParticles->begin(); it != genParticles->end(); ++it) {
    std::unique_ptr<reco::GenParticle> genpart = std::unique_ptr<reco::GenParticle>(new reco::GenParticle(*it));
    if (genpart->status() != 3) continue;
    if (fabs(genpart->pdgId()) != 13) continue;
    if (genpart->eta() < 0.0)      { GEN_muons_neg.push_back(std::move(genpart)); }
    else if (genpart->eta() > 0.0) { GEN_muons_pos.push_back(std::move(genpart)); }
    else {}
  }
  */
  // working with reco::GenParticle seems not to work anymore
  // compilation crash:
  // undefined reference to `reco::GenParticle::~GenParticle()'
  // undefined reference to `vtable for reco::GenParticle'
  // problem can be avoided by working with HepMC::GenEvent


  // const HepMC::GenEvent * myGenEvent = new HepMC::GenEvent(*(hepmcevent->GetEvent())); // old style, pre c++11 std::unique_ptr<>
  std::unique_ptr<const HepMC::GenEvent> myGenEvent = std::unique_ptr<const HepMC::GenEvent>(new HepMC::GenEvent(*(hepmcevent->GetEvent())));
  for(HepMC::GenEvent::particle_const_iterator it = myGenEvent->particles_begin(); it != myGenEvent->particles_end(); ++it) {
    std::unique_ptr<HepMC::GenParticle> genpart = std::unique_ptr<HepMC::GenParticle>(new HepMC::GenParticle(*(*it)));
    if (fabs(genpart->pdg_id()) != 13) continue;
    GEN_eta->Fill(genpart->momentum().eta()); 
    GEN_phi->Fill(genpart->momentum().phi()); 
    if (genpart->momentum().eta() < 0.0)      { GEN_muons_neg.push_back(std::move(genpart)); }
    else if (genpart->momentum().eta() > 0.0) { GEN_muons_pos.push_back(std::move(genpart)); }
    else {}
    // pointer is moved into vector and does not exist anymore at this point. access the vector if needed.
  }
  // std::cout<<"Saved GenParticles :: size = "<<GEN_muons_pos.size()+GEN_muons_neg.size()<<std::endl;

  // for (std::vector<SimTrack>::const_iterator iTrack = theSimTracks.begin(); iTrack != theSimTracks.end(); ++iTrack) {
  for (edm::SimTrackContainer::const_iterator it = SimTk->begin(); it != SimTk->end(); ++it) {
    std::unique_ptr<SimTrack> simtrack = std::unique_ptr<SimTrack>(new SimTrack(*it));
    if(fabs(simtrack->type()) != 13) continue;
    SIM_eta->Fill(simtrack->momentum().eta()); 
    SIM_phi->Fill(simtrack->momentum().phi()); 
    if(simtrack->momentum().eta() < 0.0)      { SIM_muons_neg.push_back(std::move(simtrack)); }
    else if(simtrack->momentum().eta() > 0.0) { SIM_muons_pos.push_back(std::move(simtrack)); }
    else {}
    // pointer is moved into vector and does not exist anymore at this point. access the vector if needed.
  }
  // std::cout<<"Saved SimTracks :: size = "<<SIM_muons_pos.size()+SIM_muons_neg.size()<<std::endl;

  // GEM
  for (GEMSegmentCollection::const_iterator it = gemSegmentCollection->begin(); it != gemSegmentCollection->end(); ++it) {
    GEMDetId id = it->gemDetId();
    std::unique_ptr<GEMSegment> gemseg = std::unique_ptr<GEMSegment>(new GEMSegment(*it));
    if(id.region()==-1 && id.station()==1) { GE11_segs_neg.push_back(std::move(gemseg)); }
    else if(id.region()==+1 && id.station()==1) { GE11_segs_pos.push_back(std::move(gemseg)); }
    else if(id.region()==-1 && id.station()==3) { GE21_segs_neg.push_back(std::move(gemseg)); }
    else if(id.region()==+1 && id.station()==3) { GE21_segs_pos.push_back(std::move(gemseg)); }
    else {}
  }
  // std::cout<<"Saved GEMSegments :: size = "<<GE21_segs_pos.size()+GE11_segs_pos.size()+GE21_segs_pos.size()+GE21_segs_neg.size()<<std::endl;

  // CSC
  for (CSCSegmentCollection::const_iterator it = cscSegmentCollection->begin(); it!=cscSegmentCollection->end(); ++it){
    CSCDetId id = it->cscDetId();
    std::unique_ptr<CSCSegment> cscseg = std::unique_ptr<CSCSegment>(new CSCSegment(*it));
    if(id.endcap()==0 && id.station()==1)      { ME11_segs_neg.push_back(std::move(cscseg)); }
    else if(id.endcap()==1 && id.station()==1) { ME11_segs_pos.push_back(std::move(cscseg)); }
    else if(id.endcap()==0 && id.station()==2) { ME21_segs_neg.push_back(std::move(cscseg)); }
    else if(id.endcap()==1 && id.station()==2) { ME21_segs_pos.push_back(std::move(cscseg)); }
    else {}

    //    std::cout<<" CSC Segment in Event :: DetId = "<<(it->cscDetId()).rawId()<<" = "<<cscsegmentIt->cscDetId()<<" Time :: "<<cscsegmentIt->time()<<std::endl;
    //    std::cout<<" CSC Segment Details = "<<*cscsegmentIt<<" Time :: "<<cscsegmentIt->time()<<std::endl;
  }       
  // std::cout<<"Saved CSCSegments :: size = "<<ME21_segs_pos.size()+ME11_segs_pos.size()+ME21_segs_pos.size()+ME21_segs_neg.size()<<std::endl;



  // Negative Endcap
  // ---------------
  double SIM_eta_neg = 0.0, SIM_phi_neg = 0.0;
  std::cout<<" Overview along the path of the muon :: neg endcap "<<"\n"<<" ------------------------------------------------- "<<std::endl;
  // for(std::vector< std::unique_ptr<reco::GenParticle> >::const_iterator it = GEN_muons_neg.begin(); it!=GEN_muons_neg.end(); ++it) {
  for(std::vector< std::unique_ptr<HepMC::GenParticle> >::const_iterator it = GEN_muons_neg.begin(); it!=GEN_muons_neg.end(); ++it) {
    // std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdgId()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
    // std::cout<<" | eta = "<<std::setw(9)<<(*it)->eta()<<" | phi = "<<std::setw(9)<<(*it)->phi();
    // std::cout<<" | pt = "<<std::setw(9)<<(*it)->pt()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
    // std::cout<<"in the loop"<<std::endl;
    std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdg_id()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
    std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
    std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().perp()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
  }
  for(std::vector< std::unique_ptr<SimTrack> >::const_iterator it = SIM_muons_neg.begin(); it!=SIM_muons_neg.end(); ++it) {
    std::cout<<"SIM Muon: id = "<<std::setw(2)<<(*it)->type()/*<<" | index = "<<std::setw(9)<<(*it)->genpartIndex()*/;
    std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
    std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().pt()<<std::endl;
    SIM_eta_neg = (*it)->momentum().eta();
    SIM_phi_neg = (*it)->momentum().phi();
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE11_segs_neg.begin(); it!=GE11_segs_neg.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();

    std::cout <<"GE1/1 Segment:"<<std::endl;
    std::cout <<"   GEMSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;
    
    GE11_eta->Fill(segGP.eta()); 
    GE11_phi->Fill(segGP.phi()); 
    GE11_Delta_eta->Fill(SIM_eta_neg-segGP.eta());
    GE11_Delta_phi->Fill(reco::deltaPhi(SIM_phi_neg,segGP.phi()));

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto rhr = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      GEMRecHit in DetId "<<gemid<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<" with glob pos = "<<rhGP<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME11_segs_neg.begin(); it!=ME11_segs_neg.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();

    std::cout <<"ME1/1 Segment:"<<std::endl;
    std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto rhr = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE21_segs_neg.begin(); it!=GE21_segs_neg.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();

    std::cout <<"GE2/1 Segment:"<<std::endl;
    std::cout <<"   GEMSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;

    GE21_eta->Fill(segGP.eta()); 
    GE21_phi->Fill(segGP.phi()); 
    GE21_Delta_eta->Fill(SIM_eta_neg-segGP.eta());
    GE21_Delta_phi->Fill(reco::deltaPhi(SIM_phi_neg,segGP.phi()));

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto rhr = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      GEMRecHit in DetId "<<gemid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME21_segs_neg.begin(); it!=ME21_segs_neg.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();

    std::cout <<"ME2/1 Segment:"<<std::endl;
    std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto rhr = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
    }
  }
  std::cout<<"\n"<<std::endl;


  // Positive Endcap
  // ---------------
  double SIM_eta_pos = 0.0, SIM_phi_pos = 0.0;
  std::cout<<" Overview along the path of the muon :: pos endcap "<<"\n"<<" ------------------------------------------------- "<<std::endl;
  // for(std::vector< std::unique_ptr<reco::GenParticle> >::const_iterator it = GEN_muons_pos.begin(); it!=GEN_muons_pos.end(); ++it) {
  for(std::vector< std::unique_ptr<HepMC::GenParticle> >::const_iterator it = GEN_muons_pos.begin(); it!=GEN_muons_pos.end(); ++it) {
    // std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdgId()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
    // std::cout<<" | eta = "<<std::setw(9)<<(*it)->eta()<<" | phi = "<<std::setw(9)<<(*it)->phi();
    // std::cout<<" | pt = "<<std::setw(9)<<(*it)->pt()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
    // std::cout<<"in the loop"<<std::endl;
    std::cout<<"GEN Muon: id = "<<std::setw(2)<<(*it)->pdg_id()/*<<" | index = "<<std::setw(9)<<(*it)->index()*/;
    std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
    std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().perp()<<" | st = "<<std::setw(2)<<(*it)->status()<<std::endl;
  }
  for(std::vector< std::unique_ptr<SimTrack> >::const_iterator it = SIM_muons_pos.begin(); it!=SIM_muons_pos.end(); ++it) {
    std::cout<<"SIM Muon: id = "<<std::setw(2)<<(*it)->type()/*<<" | index = "<<std::setw(9)<<(*it)->genpartIndex()*/;
    std::cout<<" | eta = "<<std::setw(9)<<(*it)->momentum().eta()<<" | phi = "<<std::setw(9)<<(*it)->momentum().phi();
    std::cout<<" | pt = "<<std::setw(9)<<(*it)->momentum().pt()<<std::endl;
    SIM_eta_pos = (*it)->momentum().eta();
    SIM_phi_pos = (*it)->momentum().phi();
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE11_segs_pos.begin(); it!=GE11_segs_pos.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();

    std::cout <<"GE1/1 Segment:"<<std::endl;
    std::cout <<"   GEMSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;

    GE11_eta->Fill(segGP.eta()); 
    GE11_phi->Fill(segGP.phi()); 
    GE11_Delta_eta->Fill(SIM_eta_pos-segGP.eta());
    GE11_Delta_phi->Fill(reco::deltaPhi(SIM_phi_pos,segGP.phi()));

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto rhr = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      GEMRecHit in DetId "<<gemid<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<" with glob pos = "<<rhGP<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME11_segs_pos.begin(); it!=ME11_segs_pos.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();

    std::cout <<"ME1/1 Segment:"<<std::endl;
    std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto rhr = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<GEMSegment> >::const_iterator it = GE21_segs_pos.begin(); it!=GE21_segs_pos.end(); ++it) {
    GEMDetId id = (*it)->gemDetId();
    auto chamb = gemGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto gemrhs = (*it)->specificRecHits();

    std::cout <<"GE2/1 Segment:"<<std::endl;
    std::cout <<"   GEMSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<gemrhs.size()<<std::endl;

    GE21_eta->Fill(segGP.eta()); 
    GE21_phi->Fill(segGP.phi()); 
    GE21_Delta_eta->Fill(SIM_eta_pos-segGP.eta());
    GE21_Delta_phi->Fill(reco::deltaPhi(SIM_phi_pos,segGP.phi()));

    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto rhr = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      GEMRecHit in DetId "<<gemid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
    }
  }
  for(std::vector< std::unique_ptr<CSCSegment> >::const_iterator it = ME21_segs_pos.begin(); it!=ME21_segs_pos.end(); ++it) {
    CSCDetId id = (*it)->cscDetId();
    auto chamb = cscGeom->chamber(id); 
    auto segLP = (*it)->localPosition();
    auto segLD = (*it)->localDirection();
    auto segGP = chamb->toGlobal(segLP);
    auto segGD = chamb->toGlobal(segLD);
    auto cscrhs = (*it)->specificRecHits();

    std::cout <<"ME2/1 Segment:"<<std::endl;
    std::cout <<"   CSCSegmnt in DetId "<<id<<" eta = "<<std::setw(9)<<segGP.eta()<<" phi = "<<std::setw(9)<<segGP.phi()<<" with glob pos = "<<segGP<<" and glob dir = "<<segGD<<std::endl;
    std::cout <<"                chi2  "<<(*it)->chi2()<<" ndof = "<<(*it)->degreesOfFreedom()<<" ==> chi2/ndof = "<<(*it)->chi2()*1.0/(*it)->degreesOfFreedom();
    std::cout << "   Number of RecHits "<<cscrhs.size()<<std::endl;

    for (auto rh = cscrhs.begin(); rh!= cscrhs.end(); rh++){
      auto cscid = rh->cscDetId();
      auto rhr = cscGeom->chamber(cscid);
      auto rhLP = rh->localPosition();
      auto rhGP = rhr->toGlobal(rhLP);
      std::cout <<"      CSCRecHit in DetId "<<cscid<<" with locl pos = "<<rhLP<<" and glob pos = "<<rhGP<<" eta = "<<std::setw(9)<<rhGP.eta()<<" phi = "<<std::setw(9)<<rhGP.phi()<<std::endl;
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGEMSegmentAnalyzer);
