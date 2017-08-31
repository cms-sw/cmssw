#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

class testPackedCandidate : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPackedCandidate);

  CPPUNIT_TEST(testDefaultConstructor);
  CPPUNIT_TEST(testCopyConstructor);
  CPPUNIT_TEST(testPackUnpack);
  CPPUNIT_TEST(testSimulateReadFromRoot);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void testDefaultConstructor() ;
  void testCopyConstructor();
  void testPackUnpack();
  void testSimulateReadFromRoot();


private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(testPackedCandidate);



void testPackedCandidate::testDefaultConstructor() {

  pat::PackedCandidate pc;

  CPPUNIT_ASSERT(pc.polarP4() == pat::PackedCandidate::PolarLorentzVector(0,0,0,0));
  CPPUNIT_ASSERT(pc.p4() == pat::PackedCandidate::LorentzVector(0,0,0,0) );
  CPPUNIT_ASSERT(pc.vertex() == pat::PackedCandidate::Point(0,0,0));
}

void testPackedCandidate::testCopyConstructor() {
  pat::PackedCandidate::LorentzVector lv(1.,0.5,0., std::sqrt(1.+0.25 +0.120*0.120));
  pat::PackedCandidate::PolarLorentzVector plv(lv.Pt(), lv.Eta(), lv.Phi(), lv.M());

  pat::PackedCandidate::Point v(0.01,0.02,0.);

  //invalid Refs use a special key
  pat::PackedCandidate pc(lv, v, 1., 1., 1., 11, reco::VertexRefProd(), reco::VertexRef().key());

  //these by design do not work
  //  CPPUNIT_ASSERT(pc.polarP4() == plv);
  //  CPPUNIT_ASSERT(pc.p4() == lv);
  //  CPPUNIT_ASSERT(pc.vertex() == v);

  pat::PackedCandidate copy_pc(pc);

  //CPPUNIT_ASSERT(copy_pc.polarP4() == plv);
  //CPPUNIT_ASSERT(copy_pc.p4() == lv);
  //CPPUNIT_ASSERT(copy_pc.vertex() == v);

  CPPUNIT_ASSERT(&copy_pc.polarP4() != &pc.polarP4());
  CPPUNIT_ASSERT(&copy_pc.p4() != &pc.p4());
  CPPUNIT_ASSERT(&copy_pc.vertex() != &pc.vertex());

}

static bool tolerance(double iLHS, double iRHS, double fraction) {
  return std::abs(iLHS-iRHS) <= fraction*std::abs(iLHS+iRHS)/2.;
}

void 
testPackedCandidate::testPackUnpack() {

  pat::PackedCandidate::LorentzVector lv(1.,1.,0., std::sqrt(2.+0.120*0.120));
  pat::PackedCandidate::PolarLorentzVector plv(lv.Pt(), lv.Eta(), lv.Phi(), lv.M());

  pat::PackedCandidate::Point v(-0.005,0.005,0.1); 
  float trkPt=plv.Pt()+0.5;
  float trkEta=plv.Eta()-0.1;
  float trkPhi=-3./4.*3.1416;


  //invalid Refs use a special key
  pat::PackedCandidate pc(lv, v, trkPt,trkEta,trkPhi, 11, reco::VertexRefProd(), reco::VertexRef().key());

  pc.pack(true);
  pc.packVtx(true);

  CPPUNIT_ASSERT(tolerance(pc.polarP4().Pt(),plv.Pt(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Eta(),plv.Eta(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Phi(),plv.Phi(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.polarP4().M(),plv.M(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.p4().X(),lv.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Y(),lv.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Z(),lv.Z(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().E(),lv.E(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.vertex().X(),v.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.vertex().Y(),v.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.vertex().Z(),v.Z(), 0.01));  
//AR : this cannot be called unless track details are set
//  CPPUNIT_ASSERT(tolerance(pc.pseudoTrack().pt(),trkPt,0.001));
//  CPPUNIT_ASSERT(tolerance(pc.pseudoTrack().eta(),trkEta,0.001));
//  CPPUNIT_ASSERT(tolerance(pc.pseudoTrack().phi(),trkPhi,0.001));
  CPPUNIT_ASSERT(tolerance(pc.ptTrk(),trkPt,0.001));
  CPPUNIT_ASSERT(tolerance(pc.etaAtVtx(),trkEta,0.001));
  CPPUNIT_ASSERT(tolerance(pc.phiAtVtx(),trkPhi,0.001));
}

void testPackedCandidate::testSimulateReadFromRoot() {

  

  pat::PackedCandidate::LorentzVector lv(1.,1.,0., std::sqrt(2. +0.120*0.120));
  pat::PackedCandidate::PolarLorentzVector plv(lv.Pt(), lv.Eta(), lv.Phi(), lv.M());
  pat::PackedCandidate::Point v(-0.005,0.005,0.1);

  float trkPt=plv.Pt()+0.5;
  float trkEta=plv.Eta()-0.1;
  float trkPhi=-3./4.*3.1416;


  //invalid Refs use a special key
  pat::PackedCandidate pc(lv, v, trkPt,trkEta,trkPhi, 11, reco::VertexRefProd(), reco::VertexRef().key());

  //  CPPUNIT_ASSERT(pc.polarP4() == plv);
  //  CPPUNIT_ASSERT(pc.p4() == lv);
  //  CPPUNIT_ASSERT(pc.vertex() == v);
  //  CPPUNIT_ASSERT(pc.pseudoTrack().p() == lv.P());

  //When reading back from ROOT, these were not stored and are nulled out
  delete pc.p4_.exchange(nullptr);
  delete pc.p4c_.exchange(nullptr);
  delete pc.vertex_.exchange(nullptr);
  delete pc.track_.exchange(nullptr);
  delete pc.m_.exchange(nullptr);

  CPPUNIT_ASSERT(tolerance(pc.polarP4().Pt(),plv.Pt(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Eta(),plv.Eta(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Phi(),plv.Phi(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.polarP4().M(),plv.M(),0.001) );
  CPPUNIT_ASSERT(tolerance(pc.p4().X(),lv.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Y(),lv.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Z(),lv.Z(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().E(),lv.E(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.vertex().X(),v.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.vertex().Y(),v.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.vertex().Z(),v.Z(), 0.01));
//AR : this cannot be called unless track details are set
//  CPPUNIT_ASSERT(tolerance(pc.pseudoTrack().pt(),trkPt,0.001));
//  CPPUNIT_ASSERT(tolerance(pc.pseudoTrack().eta(),trkEta,0.001));
//  CPPUNIT_ASSERT(tolerance(pc.pseudoTrack().phi(),trkPhi,0.001));
  CPPUNIT_ASSERT(tolerance(pc.ptTrk(),trkPt,0.001));
  CPPUNIT_ASSERT(tolerance(pc.etaAtVtx(),trkEta,0.001));
  CPPUNIT_ASSERT(tolerance(pc.phiAtVtx(),trkPhi,0.001));
  
}

