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

#include <DataFormats/GEMRecHit/interface/GEMSegmentCollection.h>
 
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>
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

  edm::EDGetTokenT<GEMSegmentCollection> GEMSegment_Token;

  std::string rootFileName;
  std::unique_ptr<TFile> outputfile;
  std::unique_ptr<TH1F> GE11_numhits;
  std::unique_ptr<TH1F> GE11_fitchi2;
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
  GEMSegment_Token = consumes<GEMSegmentCollection>(edm::InputTag("gemSegments"));

  rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");
  outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));

  GE11_fitchi2 = std::unique_ptr<TH1F>(new TH1F("chi2Vsndf","chi2Vsndf",50,0.,5.)); 
  GE11_numhits = std::unique_ptr<TH1F>(new TH1F("NumberOfHits","NumberOfHits",11,0.,10.)); 
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

  GE21_fitchi2 = std::unique_ptr<TH1F>(new TH1F("chi2Vsndf","chi2Vsndf",50,0.,5.)); 
  GE21_numhits = std::unique_ptr<TH1F>(new TH1F("NumberOfHits","NumberOfHits",11,0.,10.)); 
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
  GE11_fitchi2->Write();
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

  // ================
  // GEM Segments
  // ================
  edm::Handle<GEMSegmentCollection> gemSegment;
  iEvent.getByToken(GEMSegment_Token, gemSegment);

  std::cout <<"Number of Segments "<<gemSegment->size()<<std::endl;
  for (auto gems = gemSegment->begin(); gems != gemSegment->end(); gems++) {
    // The GEM Ensemble DetId refers to layer = 1   
    GEMDetId id = gems->gemDetId();
    std::cout <<"   Original GEMDetID "<<id<<std::endl;
    auto roll = gemGeom->etaPartition(id); 
    std::cout <<"   Global Segment Position "<<  roll->toGlobal(gems->localPosition())<<std::endl;
    auto segLP = gems->localPosition();
    auto segLD = gems->localDirection();
    std::cout <<"   Global Direction theta = "<<segLD.theta()<<" phi="<<segLD.phi()<<std::endl;
    std::cout <<"   Chi2 = "<<gems->chi2()<<" ndof = "<<gems->degreesOfFreedom()<<" ==> chi2/ndof = "<<gems->chi2()*1.0/gems->degreesOfFreedom()<<std::endl;

    auto gemrhs = gems->specificRecHits();
    std::cout <<"   GEM Ensemble Det Id "<<id<<"  Number of RecHits "<<gemrhs.size()<<std::endl;
    //loop on rechits.... take layer local position -> global -> ensemble local position same frame as segment
    for (auto rh = gemrhs.begin(); rh!= gemrhs.end(); rh++){
      auto gemid = rh->gemId();
      auto rhr = gemGeom->etaPartition(gemid);
      auto rhLP = rh->localPosition();
      auto erhLEP = rh->localPositionError();
      auto rhGP = rhr->toGlobal(rhLP); 
      auto rhLPSegm = roll->toLocal(rhGP);
      float xe  = segLP.x()+segLD.x()*rhLPSegm.z()/segLD.z();
      float ye  = segLP.y()+segLD.y()*rhLPSegm.z()/segLD.z();
      float ze = rhLPSegm.z();
      LocalPoint extrPoint(xe,ye,ze); // in segment rest frame
      auto extSegm = rhr->toLocal(roll->toGlobal(extrPoint)); // in layer restframe
      std::cout <<"      GEM Layer Id "<<rh->gemId()<<"  error on the local point "<<  erhLEP
		<<"\n-> Ensemble Rest Frame  RH local  position "<<rhLPSegm<<"  Segment extrapolation "<<extrPoint
		<<"\n-> Layer Rest Frame  RH local  position "<<rhLP<<"  Segment extrapolation "<<extSegm
		<<std::endl;

      if(gemid.station()==1) {
	GE11_fitchi2->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
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
      if(gemid.station()==2 || gemid.station()==3) {
	GE21_fitchi2->Fill(gems->chi2()*1.0/gems->degreesOfFreedom());
        GE21_numhits->Fill(gems->nRecHits());
	GE21_Residuals_x->Fill(rhLP.x()-extSegm.x());
	GE21_Residuals_y->Fill(rhLP.y()-extSegm.y());
	GE21_Pull_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	GE21_Pull_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
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
	case 3:
	  GE21_Residuals_l3_x->Fill(rhLP.x()-extSegm.x());
	  GE21_Residuals_l3_y->Fill(rhLP.y()-extSegm.y());
	  GE21_Pull_l3_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	  GE21_Pull_l3_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	  break;
	case 4:
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
    std::cout<<"\n"<<std::endl;
  }
  std::cout<<"\n"<<std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGEMSegmentAnalyzer);
