// -*- C++ -*-
//
// Package:    TestME0SegmentAnalyzer
// Class:      TestME0SegmentAnalyzer
// 
/**\class TestME0SegmentAnalyzer TestME0SegmentAnalyzer.cc MyAnalyzers/TestME0SegmentAnalyzer/src/TestME0SegmentAnalyzer.cc

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

#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>
 
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>
//
// class declaration
//

class TestME0SegmentAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestME0SegmentAnalyzer(const edm::ParameterSet&);
      ~TestME0SegmentAnalyzer();



   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
  edm::ESHandle<ME0Geometry> me0Geom;

  edm::EDGetTokenT<ME0SegmentCollection> ME0Segment_Token;

  std::string rootFileName;
  std::unique_ptr<TFile> outputfile;
  std::unique_ptr<TH1F> ME0_fitchi2;
  std::unique_ptr<TH1F> ME0_Residuals_x;
  std::unique_ptr<TH1F> ME0_Residuals_l1_x;
  std::unique_ptr<TH1F> ME0_Residuals_l2_x;
  std::unique_ptr<TH1F> ME0_Residuals_l3_x;
  std::unique_ptr<TH1F> ME0_Residuals_l4_x;
  std::unique_ptr<TH1F> ME0_Residuals_l5_x;
  std::unique_ptr<TH1F> ME0_Residuals_l6_x;
  std::unique_ptr<TH1F> ME0_Pull_x;
  std::unique_ptr<TH1F> ME0_Pull_l1_x;
  std::unique_ptr<TH1F> ME0_Pull_l2_x;
  std::unique_ptr<TH1F> ME0_Pull_l3_x;
  std::unique_ptr<TH1F> ME0_Pull_l4_x;
  std::unique_ptr<TH1F> ME0_Pull_l5_x;
  std::unique_ptr<TH1F> ME0_Pull_l6_x;
  std::unique_ptr<TH1F> ME0_Residuals_y;
  std::unique_ptr<TH1F> ME0_Residuals_l1_y;
  std::unique_ptr<TH1F> ME0_Residuals_l2_y;
  std::unique_ptr<TH1F> ME0_Residuals_l3_y;
  std::unique_ptr<TH1F> ME0_Residuals_l4_y;
  std::unique_ptr<TH1F> ME0_Residuals_l5_y;
  std::unique_ptr<TH1F> ME0_Residuals_l6_y;
  std::unique_ptr<TH1F> ME0_Pull_y;
  std::unique_ptr<TH1F> ME0_Pull_l1_y;
  std::unique_ptr<TH1F> ME0_Pull_l2_y;
  std::unique_ptr<TH1F> ME0_Pull_l3_y;
  std::unique_ptr<TH1F> ME0_Pull_l4_y;
  std::unique_ptr<TH1F> ME0_Pull_l5_y;
  std::unique_ptr<TH1F> ME0_Pull_l6_y;
};

//
// constants, enums and typedefs
//
// constructors and destructor
//
TestME0SegmentAnalyzer::TestME0SegmentAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  ME0Segment_Token = consumes<ME0SegmentCollection>(edm::InputTag("me0Segments"));

  rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");
  outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));

  ME0_fitchi2 = std::unique_ptr<TH1F>(new TH1F("chi2Vsndf","chi2Vsndf",50,0.,5.)); 
  ME0_Residuals_x    = std::unique_ptr<TH1F>(new TH1F("xME0Res","xME0Res",100,-0.5,0.5));
  ME0_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l1","xME0Res_l1",100,-0.5,0.5));
  ME0_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l2","xME0Res_l2",100,-0.5,0.5));
  ME0_Residuals_l3_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l3","xME0Res_l3",100,-0.5,0.5));
  ME0_Residuals_l4_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l4","xME0Res_l4",100,-0.5,0.5));
  ME0_Residuals_l5_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l5","xME0Res_l5",100,-0.5,0.5));
  ME0_Residuals_l6_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l6","xME0Res_l6",100,-0.5,0.5));
  ME0_Pull_x    = std::unique_ptr<TH1F>(new TH1F("xME0Pull","xME0Pull",100,-5.,5.));
  ME0_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l1","xME0Pull_l1",100,-5.,5.));
  ME0_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l2","xME0Pull_l2",100,-5.,5.));
  ME0_Pull_l3_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l3","xME0Pull_l3",100,-5.,5.));
  ME0_Pull_l4_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l4","xME0Pull_l4",100,-5.,5.));
  ME0_Pull_l5_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l5","xME0Pull_l5",100,-5.,5.));
  ME0_Pull_l6_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l6","xME0Pull_l6",100,-5.,5.));
  ME0_Residuals_y    = std::unique_ptr<TH1F>(new TH1F("yME0Res","yME0Res",100,-5.,5.));
  ME0_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l1","yME0Res_l1",100,-5.,5.));
  ME0_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l2","yME0Res_l2",100,-5.,5.));
  ME0_Residuals_l3_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l3","yME0Res_l3",100,-5.,5.));
  ME0_Residuals_l4_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l4","yME0Res_l4",100,-5.,5.));
  ME0_Residuals_l5_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l5","yME0Res_l5",100,-5.,5.));
  ME0_Residuals_l6_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l6","yME0Res_l6",100,-5.,5.));
  ME0_Pull_y    = std::unique_ptr<TH1F>(new TH1F("yME0Pull","yME0Pull",100,-5.,5.));
  ME0_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l1","yME0Pull_l1",100,-5.,5.));
  ME0_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l2","yME0Pull_l2",100,-5.,5.));
  ME0_Pull_l3_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l3","yME0Pull_l3",100,-5.,5.));
  ME0_Pull_l4_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l4","yME0Pull_l4",100,-5.,5.));
  ME0_Pull_l5_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l5","yME0Pull_l5",100,-5.,5.));
  ME0_Pull_l6_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l6","yME0Pull_l6",100,-5.,5.));
}


TestME0SegmentAnalyzer::~TestME0SegmentAnalyzer()
{
  ME0_fitchi2->Write();
  ME0_Residuals_x->Write();
  ME0_Residuals_l1_x->Write();
  ME0_Residuals_l2_x->Write();
  ME0_Residuals_l3_x->Write();
  ME0_Residuals_l4_x->Write();
  ME0_Residuals_l5_x->Write();
  ME0_Residuals_l6_x->Write();
  ME0_Pull_x->Write();
  ME0_Pull_l1_x->Write();
  ME0_Pull_l2_x->Write();
  ME0_Pull_l3_x->Write();
  ME0_Pull_l4_x->Write();
  ME0_Pull_l5_x->Write();
  ME0_Pull_l6_x->Write();
  ME0_Residuals_y->Write();
  ME0_Residuals_l1_y->Write();
  ME0_Residuals_l2_y->Write();
  ME0_Residuals_l3_y->Write();
  ME0_Residuals_l4_y->Write();
  ME0_Residuals_l5_y->Write();
  ME0_Residuals_l6_y->Write();
  ME0_Pull_y->Write();
  ME0_Pull_l1_y->Write();
  ME0_Pull_l2_y->Write();
  ME0_Pull_l3_y->Write();
  ME0_Pull_l4_y->Write();
  ME0_Pull_l5_y->Write();
  ME0_Pull_l6_y->Write();
}


//
// member functions
//

// ------------ method called for each event  ------------
void
TestME0SegmentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  iSetup.get<MuonGeometryRecord>().get(me0Geom);

  // ================
  // ME0 Segments
  // ================
  edm::Handle<ME0SegmentCollection> me0Segment;
  iEvent.getByToken(ME0Segment_Token, me0Segment);

  std::cout <<"Number of Segments "<<me0Segment->size()<<std::endl;
  for (auto me0s = me0Segment->begin(); me0s != me0Segment->end(); me0s++) {
    // The ME0 Ensemble DetId refers to layer = 1   
    ME0DetId id = me0s->me0DetId();
    std::cout <<"   Original ME0DetID "<<id<<std::endl;
    auto roll = me0Geom->etaPartition(id); 
    std::cout <<"   Global Segment Position "<<  roll->toGlobal(me0s->localPosition())<<std::endl;
    auto segLP = me0s->localPosition();
    auto segLD = me0s->localDirection();
    std::cout <<"   Global Direction theta = "<<segLD.theta()<<" phi="<<segLD.phi()<<std::endl;
    ME0_fitchi2->Fill(me0s->chi2()*1.0/me0s->degreesOfFreedom());
    std::cout <<"   Chi2 = "<<me0s->chi2()<<" ndof = "<<me0s->degreesOfFreedom()<<" ==> chi2/ndof = "<<me0s->chi2()*1.0/me0s->degreesOfFreedom()<<std::endl;

    auto me0rhs = me0s->specificRecHits();
    std::cout <<"   ME0 Ensemble Det Id "<<id<<"  Number of RecHits "<<me0rhs.size()<<std::endl;
    //loop on rechits.... take layer local position -> global -> ensemble local position same frame as segment
    for (auto rh = me0rhs.begin(); rh!= me0rhs.end(); rh++){
      auto me0id = rh->me0Id();
      auto rhr = me0Geom->etaPartition(me0id);
      auto rhLP = rh->localPosition();
      auto erhLEP = rh->localPositionError();
      auto rhGP = rhr->toGlobal(rhLP); 
      auto rhLPSegm = roll->toLocal(rhGP);
      float xe  = segLP.x()+segLD.x()*rhLPSegm.z()/segLD.z();
      float ye  = segLP.y()+segLD.y()*rhLPSegm.z()/segLD.z();
      float ze = rhLPSegm.z();
      LocalPoint extrPoint(xe,ye,ze); // in segment rest frame
      auto extSegm = rhr->toLocal(roll->toGlobal(extrPoint)); // in layer restframe
      std::cout <<"      ME0 Layer Id "<<rh->me0Id()<<"  error on the local point "<<  erhLEP
		<<"\n-> Ensemble Rest Frame  RH local  position "<<rhLPSegm<<"  Segment extrapolation "<<extrPoint
		<<"\n-> Layer Rest Frame  RH local  position "<<rhLP<<"  Segment extrapolation "<<extSegm
		<<std::endl;
      ME0_Residuals_x->Fill(rhLP.x()-extSegm.x());
      ME0_Residuals_y->Fill(rhLP.y()-extSegm.y());
      ME0_Pull_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
      ME0_Pull_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
      switch (me0id.layer()){
      case 1:
	ME0_Residuals_l1_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l1_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pull_l1_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pull_l1_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 2:
	ME0_Residuals_l2_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l2_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pull_l2_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pull_l2_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 3:
	ME0_Residuals_l3_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l3_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pull_l3_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pull_l3_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 4:
	ME0_Residuals_l4_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l4_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pull_l4_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pull_l4_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 5:
	ME0_Residuals_l5_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l5_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pull_l5_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pull_l5_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 6:
	ME0_Residuals_l6_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l6_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pull_l6_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pull_l6_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      default:
	std::cout <<"      Unphysical ME0 layer "<<me0id<<std::endl;
      }
    }
    std::cout<<"\n"<<std::endl;
  }
  std::cout<<"\n"<<std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestME0SegmentAnalyzer);
