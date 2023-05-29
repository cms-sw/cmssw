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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/ME0RecHitCollection.h>
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include "DataFormats/Math/interface/deltaPhi.h"

//
// class declaration
//

class TestME0SegmentAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TestME0SegmentAnalyzer(const edm::ParameterSet&);
  ~TestME0SegmentAnalyzer();

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  // ----------member data ---------------------------
  edm::ESGetToken<ME0Geometry, MuonGeometryRecord> me0Geom_Token;

  edm::EDGetTokenT<ME0SegmentCollection> ME0Segment_Token;
  edm::EDGetTokenT<ME0RecHitCollection> ME0RecHit_Token;
  edm::EDGetTokenT<ME0DigiPreRecoCollection> ME0Digi_Token;

  std::string rootFileName;

  std::unique_ptr<TFile> outputfile;

  std::unique_ptr<TH1F> ME0_recdR;
  std::unique_ptr<TH1F> ME0_recdPhi;
  std::unique_ptr<TH1F> ME0_segdR;
  std::unique_ptr<TH1F> ME0_segdPhi;

  std::unique_ptr<TH1F> ME0_fitchi2;
  std::unique_ptr<TH1F> ME0_rhmult;
  std::unique_ptr<TH1F> ME0_rhmultb;
  std::unique_ptr<TH1F> ME0_sgmult;
  std::unique_ptr<TH1F> ME0_shtime;
  std::unique_ptr<TH1F> ME0_rhtime;
  std::unique_ptr<TH1F> ME0_sgtime;
  std::unique_ptr<TH1F> ME0_sgterr;

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
  me0Geom_Token = esConsumes();
  ME0Segment_Token = consumes<ME0SegmentCollection>(edm::InputTag("me0Segments", "", "ME0RERECO"));
  ME0RecHit_Token = consumes<ME0RecHitCollection>(edm::InputTag("me0RecHits"));
  ME0Digi_Token = consumes<ME0DigiPreRecoCollection>(edm::InputTag("simMuonME0PseudoReDigis"));
  //  ME0SimHit_Token =
  rootFileName = iConfig.getUntrackedParameter<std::string>("RootFileName");
  outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));

  ME0_recdR = std::unique_ptr<TH1F>(new TH1F("rechitdR", "rechidR", 50, -10., 10.));
  ME0_recdPhi = std::unique_ptr<TH1F>(new TH1F("rechitdphi", "rechidphi", 50, -0.005, 0.005));
  ME0_segdR = std::unique_ptr<TH1F>(new TH1F("segmentdR", "segmentdR", 50, -10., 10.));
  ME0_segdPhi = std::unique_ptr<TH1F>(new TH1F("segmentdphi", "segmentdphi", 50, -0.1, 0.1));
  ME0_fitchi2 = std::unique_ptr<TH1F>(new TH1F("chi2Vsndf", "chi2Vsndf", 50, 0., 100.));
  ME0_rhmult = std::unique_ptr<TH1F>(new TH1F("rhmulti", "rhmulti", 11, -0.5, 10.5));
  ME0_rhmultb = std::unique_ptr<TH1F>(new TH1F("rhmultib", "rhmultib", 11, -0.5, 10.5));
  ME0_sgmult = std::unique_ptr<TH1F>(new TH1F("sgmult", "sgmult", 11, -0.5, 10.5));
  ME0_rhtime = std::unique_ptr<TH1F>(new TH1F("rhtime", "rhtime", 100, -125., 125.));
  ME0_sgtime = std::unique_ptr<TH1F>(new TH1F("sgtime", "sgtime", 100, -125., 125.));
  ME0_sgterr = std::unique_ptr<TH1F>(new TH1F("sgterr", "sgterr", 100, 0., 10.));
  ME0_Residuals_x = std::unique_ptr<TH1F>(new TH1F("xME0Res", "xME0Res", 100, -0.5, 0.5));
  ME0_Residuals_l1_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l1", "xME0Res_l1", 100, -0.5, 0.5));
  ME0_Residuals_l2_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l2", "xME0Res_l2", 100, -0.5, 0.5));
  ME0_Residuals_l3_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l3", "xME0Res_l3", 100, -0.5, 0.5));
  ME0_Residuals_l4_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l4", "xME0Res_l4", 100, -0.5, 0.5));
  ME0_Residuals_l5_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l5", "xME0Res_l5", 100, -0.5, 0.5));
  ME0_Residuals_l6_x = std::unique_ptr<TH1F>(new TH1F("xME0Res_l6", "xME0Res_l6", 100, -0.5, 0.5));
  ME0_Pull_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull", "xME0Pull", 100, -5., 5.));
  ME0_Pull_l1_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l1", "xME0Pull_l1", 100, -5., 5.));
  ME0_Pull_l2_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l2", "xME0Pull_l2", 100, -5., 5.));
  ME0_Pull_l3_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l3", "xME0Pull_l3", 100, -5., 5.));
  ME0_Pull_l4_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l4", "xME0Pull_l4", 100, -5., 5.));
  ME0_Pull_l5_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l5", "xME0Pull_l5", 100, -5., 5.));
  ME0_Pull_l6_x = std::unique_ptr<TH1F>(new TH1F("xME0Pull_l6", "xME0Pull_l6", 100, -5., 5.));
  ME0_Residuals_y = std::unique_ptr<TH1F>(new TH1F("yME0Res", "yME0Res", 100, -5., 5.));
  ME0_Residuals_l1_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l1", "yME0Res_l1", 100, -5., 5.));
  ME0_Residuals_l2_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l2", "yME0Res_l2", 100, -5., 5.));
  ME0_Residuals_l3_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l3", "yME0Res_l3", 100, -5., 5.));
  ME0_Residuals_l4_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l4", "yME0Res_l4", 100, -5., 5.));
  ME0_Residuals_l5_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l5", "yME0Res_l5", 100, -5., 5.));
  ME0_Residuals_l6_y = std::unique_ptr<TH1F>(new TH1F("yME0Res_l6", "yME0Res_l6", 100, -5., 5.));
  ME0_Pull_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull", "yME0Pull", 100, -5., 5.));
  ME0_Pull_l1_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l1", "yME0Pull_l1", 100, -5., 5.));
  ME0_Pull_l2_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l2", "yME0Pull_l2", 100, -5., 5.));
  ME0_Pull_l3_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l3", "yME0Pull_l3", 100, -5., 5.));
  ME0_Pull_l4_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l4", "yME0Pull_l4", 100, -5., 5.));
  ME0_Pull_l5_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l5", "yME0Pull_l5", 100, -5., 5.));
  ME0_Pull_l6_y = std::unique_ptr<TH1F>(new TH1F("yME0Pull_l6", "yME0Pull_l6", 100, -5., 5.));
}

TestME0SegmentAnalyzer::~TestME0SegmentAnalyzer() {
  ME0_recdR->Write();
  ME0_recdPhi->Write();
  ME0_segdR->Write();
  ME0_segdPhi->Write();
  ME0_fitchi2->Write();
  ME0_rhmult->Write();
  ME0_rhmultb->Write();
  ME0_sgmult->Write();
  ME0_rhtime->Write();
  ME0_sgtime->Write();
  ME0_sgterr->Write();
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
void TestME0SegmentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto me0Geom = iSetup.getHandle(me0Geom_Token);

  // ================
  // ME0 Segments
  // ================
  edm::Handle<ME0SegmentCollection> me0Segment;
  iEvent.getByToken(ME0Segment_Token, me0Segment);
  edm::Handle<ME0RecHitCollection> me0RecHit;
  iEvent.getByToken(ME0RecHit_Token, me0RecHit);
  edm::Handle<ME0DigiPreRecoCollection> me0Digi;
  iEvent.getByToken(ME0Digi_Token, me0Digi);

  //  std::cout <<"Numner of digi "<<me0Digi->size()<<std::endl;
  ME0DigiPreRecoCollection::DigiRangeIterator me0dgIt;
  for (me0dgIt = me0Digi->begin(); me0dgIt != me0Digi->end(); ++me0dgIt) {
    const ME0DetId me0Id = (*me0dgIt).first;
    const ME0DigiPreRecoCollection::Range& digiRange = (*me0dgIt).second;
    std::cout << " Original DIGI DET ID " << me0Id << " # digis = " << (digiRange.second - digiRange.first + 1)
              << std::endl;

    // Get the iterators over the digis associated with this LayerId
    for (ME0DigiPreRecoCollection::const_iterator digi = digiRange.first; digi != digiRange.second; digi++) {
      std::cout << "x= " << digi->x() << " , y= " << digi->y() << " timing " << digi->tof() << std::endl;
    }
  }

  std::cout << "Number of rec hit " << me0RecHit->size() << std::endl;
  float rlmax = 0.;
  float rlmin = 999999.;
  float plmin = 999999.;
  float plmax = 999999.;
  int lmin = 100;
  int lmax = 0;
  for (auto recHit = me0RecHit->begin(); recHit != me0RecHit->end(); recHit++) {
    ME0DetId id = recHit->me0Id();
    auto roll = me0Geom->etaPartition(id);
    std::cout << "   Original ME0DetID " << id << " Timing " << recHit->tof() << std::endl;
    int layer = id.layer();
    if (layer < lmin) {
      lmin = layer;
      rlmin = (roll->toGlobal(recHit->localPosition())).perp();
      plmin = (roll->toGlobal(recHit->localPosition())).barePhi();
    }
    if (layer > lmax) {
      lmax = layer;
      rlmax = (roll->toGlobal(recHit->localPosition())).perp();
      plmax = (roll->toGlobal(recHit->localPosition())).barePhi();
    }
  }
  std::cout << " Radius  max min  and delta " << rlmax << "  " << rlmin << " " << rlmax - rlmin << std::endl;
  std::cout << " Phi     max min  and delta " << plmax << "  " << plmin << " " << plmax - plmin << std::endl;
  ME0_recdR->Fill(rlmax - rlmin);
  ME0_recdPhi->Fill(fabs(deltaPhi(plmax, plmin)));

  std::cout << "Number of Segments " << me0Segment->size() << std::endl;
  ME0_sgmult->Fill(me0Segment->size());
  float hmax = 0;
  for (auto me0s = me0Segment->begin(); me0s != me0Segment->end(); me0s++) {
    // The ME0 Ensemble DetId refers to layer = 1   and roll = 1
    ME0DetId id = me0s->me0DetId();

    ME0_sgtime->Fill(me0s->time());
    ME0_sgterr->Fill(me0s->timeErr());
    std::cout << "   Original ME0DetID " << id << std::endl;
    auto roll = me0Geom->etaPartition(id);
    std::cout << "   Global Segment Position " << roll->toGlobal(me0s->localPosition()) << std::endl;
    auto segLP = me0s->localPosition();
    auto segLD = me0s->localDirection();
    std::cout << "   Local Direction theta = " << segLD.theta() << " phi=" << segLD.phi() << std::endl;
    ME0_fitchi2->Fill(me0s->chi2() * 1.0 / me0s->degreesOfFreedom());
    std::cout << "   Chi2 = " << me0s->chi2() << " ndof = " << me0s->degreesOfFreedom()
              << " ==> chi2/ndof = " << me0s->chi2() * 1.0 / me0s->degreesOfFreedom() << " Timing " << me0s->time()
              << " +- " << me0s->timeErr() << std::endl;

    auto me0rhs = me0s->specificRecHits();
    std::cout << "   ME0 Ensemble Det Id " << id << "  Number of RecHits " << me0rhs.size() << std::endl;
    ME0_rhmult->Fill(me0rhs.size());
    if (me0rhs.size() > hmax)
      hmax = me0rhs.size();
    //loop on rechits.... take layer local position -> global -> ensemble local position same frame as segment
    for (auto rh = me0rhs.begin(); rh != me0rhs.end(); rh++) {
      auto me0id = rh->me0Id();
      ME0_rhtime->Fill(rh->tof());
      auto rhr = me0Geom->etaPartition(me0id);
      auto rhLP = rh->localPosition();
      auto erhLEP = rh->localPositionError();
      auto rhGP = rhr->toGlobal(rhLP);
      auto rhLPSegm = roll->toLocal(rhGP);
      float xe = segLP.x() + segLD.x() * (rhLPSegm.z() - segLP.z()) / segLD.z();
      float ye = segLP.y() + segLD.y() * (rhLPSegm.z() - segLP.z()) / segLD.z();
      float ze = rhLPSegm.z();
      LocalPoint extrPoint(xe, ye, ze);                        // in segment rest frame
      auto extSegm = rhr->toLocal(roll->toGlobal(extrPoint));  // in layer restframe
      std::cout << "      ME0 Layer Id " << rh->me0Id() << "  error on the local point " << erhLEP
                << "\n-> Ensemble Rest Frame  RH local  position " << rhLPSegm << "  Segment extrapolation "
                << extrPoint << "\n-> Layer Rest Frame  RH local  position " << rhLP << "  Segment extrapolation "
                << extSegm << "\n-> Global Position rechit " << rhGP << " Segm Extrapolation "
                << roll->toGlobal(extrPoint) << "\n-> Timing " << rh->tof() << std::endl;
      ME0_Residuals_x->Fill(rhLP.x() - extSegm.x());
      ME0_Residuals_y->Fill(rhLP.y() - extSegm.y());
      ME0_Pull_x->Fill((rhLP.x() - extSegm.x()) / sqrt(erhLEP.xx()));
      ME0_Pull_y->Fill((rhLP.y() - extSegm.y()) / sqrt(erhLEP.yy()));
      switch (me0id.layer()) {
        case 1:
          ME0_Residuals_l1_x->Fill(rhLP.x() - extSegm.x());
          ME0_Residuals_l1_y->Fill(rhLP.y() - extSegm.y());
          ME0_Pull_l1_x->Fill((rhLP.x() - extSegm.x()) / sqrt(erhLEP.xx()));
          ME0_Pull_l1_y->Fill((rhLP.y() - extSegm.y()) / sqrt(erhLEP.yy()));
          break;
        case 2:
          ME0_Residuals_l2_x->Fill(rhLP.x() - extSegm.x());
          ME0_Residuals_l2_y->Fill(rhLP.y() - extSegm.y());
          ME0_Pull_l2_x->Fill((rhLP.x() - extSegm.x()) / sqrt(erhLEP.xx()));
          ME0_Pull_l2_y->Fill((rhLP.y() - extSegm.y()) / sqrt(erhLEP.yy()));
          break;
        case 3:
          ME0_Residuals_l3_x->Fill(rhLP.x() - extSegm.x());
          ME0_Residuals_l3_y->Fill(rhLP.y() - extSegm.y());
          ME0_Pull_l3_x->Fill((rhLP.x() - extSegm.x()) / sqrt(erhLEP.xx()));
          ME0_Pull_l3_y->Fill((rhLP.y() - extSegm.y()) / sqrt(erhLEP.yy()));
          break;
        case 4:
          ME0_Residuals_l4_x->Fill(rhLP.x() - extSegm.x());
          ME0_Residuals_l4_y->Fill(rhLP.y() - extSegm.y());
          ME0_Pull_l4_x->Fill((rhLP.x() - extSegm.x()) / sqrt(erhLEP.xx()));
          ME0_Pull_l4_y->Fill((rhLP.y() - extSegm.y()) / sqrt(erhLEP.yy()));
          break;
        case 5:
          ME0_Residuals_l5_x->Fill(rhLP.x() - extSegm.x());
          ME0_Residuals_l5_y->Fill(rhLP.y() - extSegm.y());
          ME0_Pull_l5_x->Fill((rhLP.x() - extSegm.x()) / sqrt(erhLEP.xx()));
          ME0_Pull_l5_y->Fill((rhLP.y() - extSegm.y()) / sqrt(erhLEP.yy()));
          break;
        case 6:
          ME0_Residuals_l6_x->Fill(rhLP.x() - extSegm.x());
          ME0_Residuals_l6_y->Fill(rhLP.y() - extSegm.y());
          ME0_Pull_l6_x->Fill((rhLP.x() - extSegm.x()) / sqrt(erhLEP.xx()));
          ME0_Pull_l6_y->Fill((rhLP.y() - extSegm.y()) / sqrt(erhLEP.yy()));
          break;
        default:
          std::cout << "      Unphysical ME0 layer " << me0id << std::endl;
      }
    }
    std::cout << "\n" << std::endl;
  }
  std::cout << "\n" << std::endl;
  ME0_rhmultb->Fill(hmax);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestME0SegmentAnalyzer);
