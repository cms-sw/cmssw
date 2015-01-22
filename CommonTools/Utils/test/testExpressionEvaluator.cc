#include <cppunit/extensions/HelperMacros.h>
#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"


#include "CommonTools/Utils/test/ExprEvalStubs/CutOnCandidate.h"

class testExpressionEvaluator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testExpressionEvaluator);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {
   std::cerr << "setting up VITest/ExprEval" << std::endl;
   system("mkdir -p $CMSSW_BASE/src/VITest/ExprEval/src; cp CommonTools/Utils/test/ExprEvalStubs/*.h $CMSSW_BASE/src/VITest/ExprEval/src/.");
   system("cp CommonTools/Utils/test/ExprEvalStubs/BuildFile.xml $CMSSW_BASE/src/VITest/ExprEval/.; pushd $CMSSW_BASE; scram b -j 8; popd");
  }
  void tearDown() {system("rm -rf $CMSSW_BASE/src/VITest");}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION( testExpressionEvaluator );

#include <iostream>
namespace {
  void checkCandidate(reco::Candidate const & cand, const std::string & expression, double x) {
    std::cerr << "testing " << expression << std::endl;
    try {
     std::string sexpr = "double eval(reco::Candidate const& cand) const override { return ";
     sexpr += expression + ";}";
     reco::ExpressionEvaluator parser("VITest/ExprEval","eetest::ValueOnCandidate",sexpr.c_str());
     auto expr = parser.expr<eetest::ValueOnCandidate>();
     CPPUNIT_ASSERT(std::abs(expr->eval(cand) - x) < 1.e-6);
    } catch(cms::Exception const & e) {
      std::cerr << e.what()  << std::endl;
      CPPUNIT_ASSERT("ExpressionEvaluator threw"==0);
    }
  }
}

void testExpressionEvaluator::checkAll() {

  reco::CompositeCandidate cand;

  reco::Candidate::LorentzVector p1(1, 2, 3, 4);
  reco::Candidate::LorentzVector p2(1.1, -2.5, 4.3, 13.7);
  reco::LeafCandidate c1(+1, p1);
  reco::LeafCandidate c2(-1, p2);
  cand.addDaughter(c1);
  cand.addDaughter(c2);
  CPPUNIT_ASSERT(cand.numberOfDaughters()==2);
  CPPUNIT_ASSERT(cand.daughter(0)!=0);
  CPPUNIT_ASSERT(cand.daughter(1)!=0);
  {
    checkCandidate(cand,"cand.numberOfDaughters()", cand.numberOfDaughters());
    checkCandidate(cand,"cand.daughter(0)->isStandAloneMuon()", cand.daughter(0)->isStandAloneMuon());  
    checkCandidate(cand,"cand.daughter(1)->isStandAloneMuon()", cand.daughter(1)->isStandAloneMuon());  
    checkCandidate(cand,"cand.daughter(0)->pt()", cand.daughter(0)->pt());
    checkCandidate(cand,"cand.daughter(1)->pt()", cand.daughter(1)->pt());
    checkCandidate(cand,"std::min(cand.daughter(0)->pt(), cand.daughter(1)->pt())", std::min(cand.daughter(0)->pt(), cand.daughter(1)->pt()));
    checkCandidate(cand,"std::max(cand.daughter(0)->pt(), cand.daughter(1)->pt())", std::max(cand.daughter(0)->pt(), cand.daughter(1)->pt()));
    checkCandidate(cand,"reco::deltaPhi(cand.daughter(0)->phi(), cand.daughter(1)->phi())", reco::deltaPhi(cand.daughter(0)->phi(), cand.daughter(1)->phi()));
    // check also opposite order, to see that the sign is correct
    checkCandidate(cand,"reco::deltaPhi(cand.daughter(1)->phi(), cand.daughter(0)->phi())", reco::deltaPhi(cand.daughter(1)->phi(), cand.daughter(0)->phi())); 
    checkCandidate(cand,"reco::deltaR(*cand.daughter(0), *cand.daughter(1))", reco::deltaR(*cand.daughter(0), *cand.daughter(1)));

  }


}
