#include <cppunit/extensions/HelperMacros.h>
#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"


class testExpressionEvaluator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testExpressionEvaluator);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {
    system("mkdir -p $CMSSW_BASE/src/VITest/ExprEval/src; cp CommonTools/Utils/test/ExprEvalStubs/*.h $CMSSW_BASE/src/VITest/ExprEval/src/.; pushd $CMSSW_BASE; scram b -j 8; popd");
  }
  void tearDown() {system("rm -rf $CMSSW_BASE/src/VITest");}
  void checkAll(); 
};


void testExpressionEvaluator::checkAll() {




}
