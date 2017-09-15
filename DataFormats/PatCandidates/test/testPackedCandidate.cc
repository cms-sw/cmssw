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
  CPPUNIT_TEST(testPackUnpackTime);

  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}

  void testDefaultConstructor() ;
  void testCopyConstructor();
  void testPackUnpack();
  void testSimulateReadFromRoot();

  void testPackUnpackTime();

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


void testPackedCandidate::testPackUnpackTime() {
  bool debug = false; // turn this on in order to get a printout of the numerical precision you get for the timing in the various encodings

  if (debug) std::cout << std::endl;
  if (debug) std::cout << "Minimum time error: " << pat::PackedCandidate::unpackTimeError(1) << std::endl;
  if (debug) std::cout << "Maximum time error: " << pat::PackedCandidate::unpackTimeError(255) << std::endl;
  float avgres = 0; int navg = 0;
  for (int i = 2; i < 255; ++i) {
    float unp = pat::PackedCandidate::unpackTimeError(i);
    float res = 0.5*(pat::PackedCandidate::unpackTimeError(i+1)-pat::PackedCandidate::unpackTimeError(i-1));
    avgres += (res/unp); navg++;
    if (debug) std::cout << " i = " << i << " unp = " << unp << " quant error = " << res << "  packed = " << int(pat::PackedCandidate::packTimeError(unp+0.3*res)) << " and " << int(pat::PackedCandidate::packTimeError(unp-0.3*res)) << std::endl;
    CPPUNIT_ASSERT(pat::PackedCandidate::packTimeError(unp+0.3*res) == i);
    CPPUNIT_ASSERT(pat::PackedCandidate::packTimeError(unp-0.3*res) == i);
  }
  if (debug) std::cout << "Average rel uncertainty: " << (avgres/navg) << std::endl;
  if (debug) std::cout << std::endl;

  if (debug) std::cout << "Zero time standalone (pos): " << pat::PackedCandidate::unpackTimeNoError(0) << std::endl;
  if (debug) std::cout << "Minimum time standalone (pos): " << pat::PackedCandidate::unpackTimeNoError(+1) << std::endl;
  if (debug) std::cout << "Minimum time standalone (neg): " << pat::PackedCandidate::unpackTimeNoError(-1) << std::endl;
  if (debug) std::cout << "Maximum time standalone, 8 bits (pos): " << pat::PackedCandidate::unpackTimeNoError(+255) << std::endl;
  if (debug) std::cout << "Maximum time standalone, 8 bits (neg): " << pat::PackedCandidate::unpackTimeNoError(-255) << std::endl;
  if (debug) std::cout << "Maximum time standalone, 10 bits (pos): " << pat::PackedCandidate::unpackTimeNoError(+1023) << std::endl;
  if (debug) std::cout << "Maximum time standalone, 10 bits (neg): " << pat::PackedCandidate::unpackTimeNoError(-1023) << std::endl;
  if (debug) std::cout << "Maximum time standalone, 11 bits (pos): " << pat::PackedCandidate::unpackTimeNoError(+2047) << std::endl;
  if (debug) std::cout << "Maximum time standalone, 11 bits (neg): " << pat::PackedCandidate::unpackTimeNoError(-2047) << std::endl;
  avgres = 0; navg = 0;
  for (int i = 2; i < 2040; i *= 1.5) {
    float unp = pat::PackedCandidate::unpackTimeNoError(i);
    float res = 0.5*(pat::PackedCandidate::unpackTimeNoError(i+1)-pat::PackedCandidate::unpackTimeNoError(i-1));
    avgres += (res/unp); navg++;
    if (debug) std::cout << " i = +" << i << " unp = +" << unp << " quant error = " << res << "  packed = " << int(pat::PackedCandidate::packTimeNoError(unp+0.3*res)) << " and +" << int(pat::PackedCandidate::packTimeNoError(unp-0.3*res)) << std::endl;
    CPPUNIT_ASSERT(pat::PackedCandidate::packTimeNoError(unp+0.3*res) == i);
    CPPUNIT_ASSERT(pat::PackedCandidate::packTimeNoError(unp-0.3*res) == i);
    unp = pat::PackedCandidate::unpackTimeNoError(-i);
    res = 0.5*(pat::PackedCandidate::unpackTimeNoError(-i+1)-pat::PackedCandidate::unpackTimeNoError(-i-1));
    avgres += std::abs(res/unp); navg++;
    if (debug) std::cout << " i = " << -i << " unp = " << unp << " quant error = " << res << "  packed = " << int(pat::PackedCandidate::packTimeNoError(unp+0.3*res)) << " and " << int(pat::PackedCandidate::packTimeNoError(unp-0.3*res)) << std::endl;
    CPPUNIT_ASSERT(pat::PackedCandidate::packTimeNoError(unp+0.3*res) == -i);
    CPPUNIT_ASSERT(pat::PackedCandidate::packTimeNoError(unp-0.3*res) == -i);
  }
  if (debug) std::cout << "Average rel uncertainty: " << (avgres/navg) << std::endl;
  if (debug) std::cout << std::endl;

  for (float aTimeErr = 2.0e-3; aTimeErr <= 1000e-3; aTimeErr *= std::sqrt(5.f)) {
      uint8_t packedTimeErr = pat::PackedCandidate::packTimeError(aTimeErr);
      float unpackedTimeErr = pat::PackedCandidate::unpackTimeError(packedTimeErr);
      if (debug) std::cout << "For a timeError of " << aTimeErr << " ns (uint8: " << unsigned(packedTimeErr) << ", unpack " << unpackedTimeErr << ")" << std::endl;
      if (debug) std::cout << "Minimum time (pos): " << pat::PackedCandidate::unpackTimeWithError(+1, packedTimeErr) << std::endl;
      if (debug) std::cout << "Minimum time (neg): " << pat::PackedCandidate::unpackTimeWithError(-1, packedTimeErr) << std::endl;
      if (debug) std::cout << "Maximum time 8 bits (pos): " << pat::PackedCandidate::unpackTimeWithError(+254, packedTimeErr) << std::endl;
      if (debug) std::cout << "Maximum time 8 bits (neg): " << pat::PackedCandidate::unpackTimeWithError(-254, packedTimeErr) << std::endl;
      if (debug) std::cout << "Maximum time 10 bits (pos): " << pat::PackedCandidate::unpackTimeWithError(+1022, packedTimeErr) << std::endl;
      if (debug) std::cout << "Maximum time 10 bits (neg): " << pat::PackedCandidate::unpackTimeWithError(-1022, packedTimeErr) << std::endl;
      if (debug) std::cout << "Maximum time 12 bits (pos): " << pat::PackedCandidate::unpackTimeWithError(+4094, packedTimeErr) << std::endl;
      if (debug) std::cout << "Maximum time 12 bits (neg): " << pat::PackedCandidate::unpackTimeWithError(-4094, packedTimeErr) << std::endl;
      avgres = 0; navg = 0;
      for (int i = 2; i < 4096; i *= 3) {
        float unp = pat::PackedCandidate::unpackTimeWithError(i,packedTimeErr);
        float res = 0.5*(pat::PackedCandidate::unpackTimeWithError(i+2,packedTimeErr)-pat::PackedCandidate::unpackTimeWithError(i-2,packedTimeErr));
        avgres += (res); navg++;
        if (debug) std::cout << " i = +" << i << " unp = +" << unp << " quant error = " << res << "  packed = +" << int(pat::PackedCandidate::packTimeWithError(unp+0.2*res, unpackedTimeErr)) << " and +" << int(pat::PackedCandidate::packTimeWithError(unp-0.2*res, unpackedTimeErr)) << std::endl;
        CPPUNIT_ASSERT(pat::PackedCandidate::packTimeWithError(unp+0.2*res,unpackedTimeErr) == i);
        CPPUNIT_ASSERT(pat::PackedCandidate::packTimeWithError(unp-0.2*res,unpackedTimeErr) == i);
        unp = pat::PackedCandidate::unpackTimeWithError(-i,packedTimeErr);
        res = 0.5*(pat::PackedCandidate::unpackTimeWithError(-i+2,packedTimeErr)-pat::PackedCandidate::unpackTimeWithError(-i-2,packedTimeErr));
        avgres += std::abs(res); navg++;
        if (debug) std::cout << " i = " << -i << " unp = " << unp << " quant error = " << res << "  packed = " << int(pat::PackedCandidate::packTimeWithError(unp+0.2*res, unpackedTimeErr)) << " and " << int(pat::PackedCandidate::packTimeWithError(unp-0.2*res, unpackedTimeErr)) << std::endl;
        CPPUNIT_ASSERT(pat::PackedCandidate::packTimeWithError(unp+0.2*res,unpackedTimeErr) == -i);
        CPPUNIT_ASSERT(pat::PackedCandidate::packTimeWithError(unp-0.2*res,unpackedTimeErr) == -i);
      }
      if (debug) std::cout << "Average abs uncertainty: " << (avgres/navg) << std::endl;
      if (debug) std::cout << "Now testing overflows: " << std::endl;
      avgres = 0; navg = 0;
      for (float aTime = aTimeErr; aTime <= 200; aTime *= std::sqrt(5.f)) {
        int i = pat::PackedCandidate::packTimeWithError(aTime,unpackedTimeErr);
        float res = 0.5*std::abs(pat::PackedCandidate::unpackTimeWithError(i+2,packedTimeErr)-pat::PackedCandidate::unpackTimeWithError(i-2,packedTimeErr));
        float unp = pat::PackedCandidate::unpackTimeWithError(i,packedTimeErr);
        if (debug) std::cout << " t = +" << aTime << " i = +" << i << "   quant error = " << res << " unpacked = +" << unp << "   diff/res = " << (aTime-unp)/res << std::endl;
        CPPUNIT_ASSERT(std::abs(unp-aTime) < res);
        avgres = std::max(avgres,std::abs(res/unp));
        i = pat::PackedCandidate::packTimeWithError(-aTime,unpackedTimeErr);
        res = 0.5*std::abs(pat::PackedCandidate::unpackTimeWithError(i+2,packedTimeErr)-pat::PackedCandidate::unpackTimeWithError(i-2,packedTimeErr));
        unp = pat::PackedCandidate::unpackTimeWithError(i,packedTimeErr);
        if (debug) std::cout << " t = " << -aTime << " i = " << i << "   quant error = " << res << " unpacked = " << unp << "   diff/res = " << (-aTime-unp)/res << std::endl;
        CPPUNIT_ASSERT(std::abs(unp+aTime) < res);
        avgres = std::max(avgres,std::abs(res/unp));
      }
      if (debug) std::cout << "Worst rel uncertainty: " << (avgres) << std::endl;
      if (debug) std::cout << std::endl;

  }  
  if (debug) std::cout << std::endl;
}

