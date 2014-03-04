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
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      //virtual void endRun(edm::Run const&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------
  edm::ESHandle<ME0Geometry> me0Geom;

  std::string rootFileName;
  TFile * outputfile;
  TH1F * ME0_fitchi2;
  TH1F * ME0_Residuals_x;
  TH1F * ME0_Residuals_l1_x;
  TH1F * ME0_Residuals_l2_x;
  TH1F * ME0_Residuals_l3_x;
  TH1F * ME0_Residuals_l4_x;
  TH1F * ME0_Residuals_l5_x;
  TH1F * ME0_Residuals_l6_x;
  TH1F * ME0_Pool_x;
  TH1F * ME0_Pool_l1_x;
  TH1F * ME0_Pool_l2_x;
  TH1F * ME0_Pool_l3_x;
  TH1F * ME0_Pool_l4_x;
  TH1F * ME0_Pool_l5_x;
  TH1F * ME0_Pool_l6_x;
  TH1F * ME0_Residuals_y;
  TH1F * ME0_Residuals_l1_y;
  TH1F * ME0_Residuals_l2_y;
  TH1F * ME0_Residuals_l3_y;
  TH1F * ME0_Residuals_l4_y;
  TH1F * ME0_Residuals_l5_y;
  TH1F * ME0_Residuals_l6_y;
  TH1F * ME0_Pool_y;
  TH1F * ME0_Pool_l1_y;
  TH1F * ME0_Pool_l2_y;
  TH1F * ME0_Pool_l3_y;
  TH1F * ME0_Pool_l4_y;
  TH1F * ME0_Pool_l5_y;
  TH1F * ME0_Pool_l6_y;
};

//
// constants, enums and typedefs
//
// constructors and destructor
//
TestME0SegmentAnalyzer::TestME0SegmentAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");

  outputfile = new TFile(rootFileName.c_str(), "RECREATE" );
  ME0_fitchi2 = new TH1F("chi2Vsndf","chi2Vsndf",20,0.,2.); 
  ME0_Residuals_x    = new TH1F("xME0Res","xME0Res",100,-0.5,0.5);
  ME0_Residuals_l1_x = new TH1F("xME0Res_l1","xME0Res_l1",100,-0.5,0.5);
  ME0_Residuals_l2_x = new TH1F("xME0Res_l2","xME0Res_l2",100,-0.5,0.5);
  ME0_Residuals_l3_x = new TH1F("xME0Res_l3","xME0Res_l3",100,-0.5,0.5);
  ME0_Residuals_l4_x = new TH1F("xME0Res_l4","xME0Res_l4",100,-0.5,0.5);
  ME0_Residuals_l5_x = new TH1F("xME0Res_l5","xME0Res_l5",100,-0.5,0.5);
  ME0_Residuals_l6_x = new TH1F("xME0Res_l6","xME0Res_l6",100,-0.5,0.5);
  ME0_Pool_x    = new TH1F("xME0Pool","xME0Pool",100,-5.,5.);
  ME0_Pool_l1_x = new TH1F("xME0Pool_l1","xME0Pool_l1",100,-5.,5.);
  ME0_Pool_l2_x = new TH1F("xME0Pool_l2","xME0Pool_l2",100,-5.,5.);
  ME0_Pool_l3_x = new TH1F("xME0Pool_l3","xME0Pool_l3",100,-5.,5.);
  ME0_Pool_l4_x = new TH1F("xME0Pool_l4","xME0Pool_l4",100,-5.,5.);
  ME0_Pool_l5_x = new TH1F("xME0Pool_l5","xME0Pool_l5",100,-5.,5.);
  ME0_Pool_l6_x = new TH1F("xME0Pool_l6","xME0Pool_l6",100,-5.,5.);
  ME0_Residuals_y    = new TH1F("yME0Res","yME0Res",100,-5.,5.);
  ME0_Residuals_l1_y = new TH1F("yME0Res_l1","yME0Res_l1",100,-5.,5.);
  ME0_Residuals_l2_y = new TH1F("yME0Res_l2","yME0Res_l2",100,-5.,5.);
  ME0_Residuals_l3_y = new TH1F("yME0Res_l3","yME0Res_l3",100,-5.,5.);
  ME0_Residuals_l4_y = new TH1F("yME0Res_l4","yME0Res_l4",100,-5.,5.);
  ME0_Residuals_l5_y = new TH1F("yME0Res_l5","yME0Res_l5",100,-5.,5.);
  ME0_Residuals_l6_y = new TH1F("yME0Res_l6","yME0Res_l6",100,-5.,5.);
  ME0_Pool_y    = new TH1F("yME0Pool","yME0Pool",100,-5.,5.);
  ME0_Pool_l1_y = new TH1F("yME0Pool_l1","yME0Pool_l1",100,-5.,5.);
  ME0_Pool_l2_y = new TH1F("yME0Pool_l2","yME0Pool_l2",100,-5.,5.);
  ME0_Pool_l3_y = new TH1F("yME0Pool_l3","yME0Pool_l3",100,-5.,5.);
  ME0_Pool_l4_y = new TH1F("yME0Pool_l4","yME0Pool_l4",100,-5.,5.);
  ME0_Pool_l5_y = new TH1F("yME0Pool_l5","yME0Pool_l5",100,-5.,5.);
  ME0_Pool_l6_y = new TH1F("yME0Pool_l6","yME0Pool_l6",100,-5.,5.);
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
  ME0_Pool_x->Write();
  ME0_Pool_l1_x->Write();
  ME0_Pool_l2_x->Write();
  ME0_Pool_l3_x->Write();
  ME0_Pool_l4_x->Write();
  ME0_Pool_l5_x->Write();
  ME0_Pool_l6_x->Write();
  ME0_Residuals_y->Write();
  ME0_Residuals_l1_y->Write();
  ME0_Residuals_l2_y->Write();
  ME0_Residuals_l3_y->Write();
  ME0_Residuals_l4_y->Write();
  ME0_Residuals_l5_y->Write();
  ME0_Residuals_l6_y->Write();
  ME0_Pool_y->Write();
  ME0_Pool_l1_y->Write();
  ME0_Pool_l2_y->Write();
  ME0_Pool_l3_y->Write();
  ME0_Pool_l4_y->Write();
  ME0_Pool_l5_y->Write();
  ME0_Pool_l6_y->Write();
}


//
// member functions
//

// ------------ method called for each event  ------------
void
TestME0SegmentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // ================
  // ME0 Segmentss
  // ================
  edm::Handle<ME0SegmentCollection> me0Segment;
  iEvent.getByLabel("me0Segments","",me0Segment);

  
  std::cout <<" NUmbero of Segments "<<me0Segment->size()<<std::endl;
  for (auto me0s = me0Segment->begin(); me0s != me0Segment->end(); me0s++) {
    // The ME0 Ensamble DetId refers to layer = 1   
    ME0DetId id = me0s->me0DetId();
    std::cout <<" Original ME0DetID "<<id<<std::endl;
    auto roll = me0Geom->etaPartition(id); 
    std::cout <<"Global Segment Position "<<  roll->toGlobal(me0s->localPosition())<<std::endl;
    auto segLP = me0s->localPosition();
    auto segLD = me0s->localDirection();
    std::cout <<" Global Direction theta = "<<segLD.theta()<<" phi="<<segLD.phi()<<std::endl;
    auto me0rhs = me0s->specificRecHits();
    std::cout <<"ME0 Ensamble Det Id "<<id<<"  Numbero of RecHits "<<me0rhs.size()<<std::endl;
    //loop on rechits.... take layer local position -> global -> ensamble local position same frame as segment
    ME0_fitchi2->Fill(me0s->chi2()/me0s->degreesOfFreedom());
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
      std::cout <<" ME0 Layer Id "<<rh->me0Id()<<"  error on the local point "<<  erhLEP
		<<"\n-> Ensamble Rest Frame  RH local  position "<<rhLPSegm<<"  Segment extrapolation "<<extrPoint
		<<"\n-> Layer Rest Frame  RH local  position "<<rhLP<<"  Segment extrapolation "<<extSegm
		<<std::endl;
      ME0_Residuals_x->Fill(rhLP.x()-extSegm.x());
      ME0_Residuals_y->Fill(rhLP.y()-extSegm.y());
      ME0_Pool_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
      ME0_Pool_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
      switch (me0id.layer()){
      case 1:
	ME0_Residuals_l1_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l1_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pool_l1_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pool_l1_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 2:
	ME0_Residuals_l2_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l2_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pool_l2_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pool_l2_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 3:
	ME0_Residuals_l3_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l3_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pool_l3_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pool_l3_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 4:
	ME0_Residuals_l4_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l4_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pool_l4_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pool_l4_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 5:
	ME0_Residuals_l5_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l5_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pool_l5_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pool_l5_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      case 6:
	ME0_Residuals_l6_x->Fill(rhLP.x()-extSegm.x());
	ME0_Residuals_l6_y->Fill(rhLP.y()-extSegm.y());
	ME0_Pool_l6_x->Fill((rhLP.x()-extSegm.x())/sqrt(erhLEP.xx()));
	ME0_Pool_l6_y->Fill((rhLP.y()-extSegm.y())/sqrt(erhLEP.yy()));
	break;
      default:
	std::cout <<" Unphysical ME0 layer "<<me0id<<std::endl;
      }
    }
  }
}
// ------------ method called once each job just before starting event loop  ------------
void 
TestME0SegmentAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestME0SegmentAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
TestME0SegmentAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(me0Geom);

}

//define this as a plug-in
DEFINE_FWK_MODULE(TestME0SegmentAnalyzer);
