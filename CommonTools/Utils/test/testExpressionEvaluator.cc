#include <cppunit/extensions/HelperMacros.h>

#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

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
  testExpressionEvaluator() {} // for crappy pats
  ~testExpressionEvaluator(){}
  void checkAll(); 

};

CPPUNIT_TEST_SUITE_REGISTRATION( testExpressionEvaluator );

#include <iostream>
namespace {
  void checkCandidate(reco::LeafCandidate const & cand, const std::string & expression, double x) {
    std::cerr << "testing " << expression << std::endl;
    try {
     //provide definition of the virtual function as a string
     std::string sexpr = "double eval(reco::LeafCandidate const& cand) const override { return ";
     sexpr += expression + ";}";
     // construct the expression evaluator (pkg where precompile.h resides, name of base class, declaration of overloaded member function)
     reco::ExpressionEvaluator eval("CommonTools/CandUtils","reco::ValueOnObject<reco::LeafCandidate>",sexpr.c_str());
     // obtain a pointer to the base class  (to be stored in Filter and Analyser at thier costruction time!)
     reco::ValueOnObject<reco::LeafCandidate> const * expr = eval.expr<reco::ValueOnObject<reco::LeafCandidate>>();
     CPPUNIT_ASSERT(expr);
     // invoke
     CPPUNIT_ASSERT(std::abs(expr->eval(cand) - x) < 1.e-6);
    } catch(cms::Exception const & e) {
      // if compilation fails, the compiler output is part of the exception message
      std::cerr << e.what()  << std::endl;
      CPPUNIT_ASSERT("ExpressionEvaluator threw"==0);
    }
  }

  std::vector<reco::LeafCandidate>  generate() {
     reco::Candidate::LorentzVector p1(10, -10, -10, 15);
     reco::Candidate::LorentzVector incr(0, 3, 3, 0);

     int sign=1;
     std::vector<reco::LeafCandidate> ret;
     for (int i=0; i<10; ++i) {
       ret.emplace_back(sign,p1);
       sign = -sign;
       p1 += incr;
     }
     return ret;
   }


  struct MyAnalyzer {
    using Selector = reco::MaskCollection<reco::LeafCandidate>;
    explicit MyAnalyzer(std::string const & cut) {
      std::string sexpr = "void eval(Collection const & c, Mask & m) const override{";
      sexpr += "\n auto cut = [](reco::LeafCandidate const & cand){ return "+cut+";};\n"; 
      sexpr += "mask(c,m,cut); }";
      std::cerr << "testing " << sexpr << std::endl;
      try {
        reco::ExpressionEvaluator eval("CommonTools/CandUtils","reco::MaskCollection<reco::LeafCandidate>",sexpr.c_str());
        m_selector = eval.expr<Selector>();
        CPPUNIT_ASSERT(m_selector);
      } catch(cms::Exception const & e) {
        std::cerr << e.what()  << std::endl;
        CPPUNIT_ASSERT("ExpressionEvaluator threw"==0);
      }

    }

    void analyze() const {
      auto inputColl = generate();
      Selector::Collection cands; cands.reserve(inputColl.size());
      for (auto const & c : inputColl) cands.push_back(&c);
      Selector::Mask mask;
      m_selector->eval(cands,mask);    
      CPPUNIT_ASSERT(2==std::count(mask.begin(),mask.end(),true));
      int ind=0;
      cands.erase(std::remove_if(cands.begin(),cands.end(),[&](Selector::Collection::value_type const &){return !mask[ind++];}),cands.end());
      CPPUNIT_ASSERT(2==cands.size());
     }


     Selector const * m_selector = nullptr;
  };


  struct MyAnalyzer2 {
    using Selector = reco::SelectInCollection<reco::LeafCandidate>;
    explicit MyAnalyzer2(std::string const & cut) {
      std::string sexpr = "void eval(Collection & c) const override{";
      sexpr += "\n auto cut = [](reco::LeafCandidate const & cand){ return "+cut+";};\n";
      sexpr += "select(c,cut); }";
      std::cerr << "testing " << sexpr << std::endl;
      try {
	reco::ExpressionEvaluator eval("CommonTools/CandUtils","reco::SelectInCollection<reco::LeafCandidate>",sexpr.c_str());
        m_selector = eval.expr<Selector>();
        CPPUNIT_ASSERT(m_selector);
      } catch(cms::Exception const & e) {
        std::cerr << e.what()  << std::endl;
        CPPUNIT_ASSERT("ExpressionEvaluator threw"==0);
      }                                                                                                     
    }

    void analyze() const {
      auto inputColl = generate();
      Selector::Collection cands; cands.reserve(inputColl.size());
      for (auto const & c : inputColl) cands.push_back(&c);
      m_selector->eval(cands);
      CPPUNIT_ASSERT(2==cands.size());
    }

    Selector const * m_selector = nullptr;
  };



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

  MyAnalyzer analyzer("cand.pt()>15 & std::abs(cand.eta())<2");
  analyzer.analyze();

  MyAnalyzer2 analyzer2("cand.pt()>15 & std::abs(cand.eta())<2");
  analyzer2.analyze();


  // pat

  std::vector<reco::LeafCandidate> cands;
  cands.push_back(c1);  cands.push_back(c2); 
  edm::TestHandle<std::vector<reco::LeafCandidate> > constituentsHandle(&cands, edm::ProductID(42));
  reco::Jet::Constituents constituents;
  constituents.push_back( reco::Jet::Constituent(constituentsHandle, 0) );
  constituents.push_back( reco::Jet::Constituent(constituentsHandle, 1) );
  reco::CaloJet::Specific caloSpecific; caloSpecific.mMaxEInEmTowers = 0.5;
  pat::Jet jet(reco::CaloJet(p1+p2, reco::Jet::Point(), caloSpecific, constituents));
  { 
  std::string expression = "jet.userData<math::XYZVector>(\"my2int\")";
  std::cerr << "testing " << expression << std::endl;
    try {
     //provide definition of the virtual function as a string
     std::string sexpr = "math::XYZVector const * operator()(pat::Jet const& jet) const override { return ";
     sexpr += expression + ";}";
     // obtain a pointer to the base class  (to be stored in Filter and Analyser at thier costruction time!)
     auto const * expr = reco_expressionEvaluator("CommonTools/RecoUtils",SINGLE_ARG(reco::genericExpression<math::XYZVector const *, pat::Jet const &>),sexpr);
     CPPUNIT_ASSERT(expr);
     // invoke
     CPPUNIT_ASSERT((*expr)(jet)==nullptr);
     jet.addUserData("my2int",math::XYZVector(-2,-1,2));
     CPPUNIT_ASSERT(jet.userData<math::XYZVector>("my2int")->z()==2);
     std::cout << "now expr eval" << std::endl;
     CPPUNIT_ASSERT((*expr)(jet)->z()==2);
    } catch(cms::Exception const & e) {
      // if compilation fails, the compiler output is part of the exception message
      std::cerr << e.what()  << std::endl;
      CPPUNIT_ASSERT("ExpressionEvaluator threw"==0);
    }
  }

}
