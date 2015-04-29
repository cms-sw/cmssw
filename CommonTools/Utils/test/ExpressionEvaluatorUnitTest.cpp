#include "CommonTools/Utils/src/CandForTest.h"

#include "CommonTools/Utils/interface/ExpressionEvaluator.h"

#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <atomic>

int main() {

   // build fake test package...
   std::string pkg = "ExpressionEvaluatorTests/EEUnitTest";

  using reco::ExpressionEvaluator;

  using vcut = reco::genericExpression<bool,int,int>;
  using Cand = eetest::CandForTest;
  using MyExpr = reco::MaskCollection<eetest::CandForTest>;

  std::vector<Cand> oriC;
  MyExpr::Collection c;
  for (int i=5; i<15; ++i) { oriC.emplace_back(Cand(i,1,1)); c.push_back(&oriC.back()); }
  MyExpr::Mask r;

  std::string expr = "void eval(Collection const & c,  Mask & r) const override{ r.resize(c.size()); std::transform(c.begin(),c.end(),r.begin(), [](Collection::value_type const & c){ return (*c).pt()>10;}); }";

  ExpressionEvaluator parser("CommonTools/Utils", "reco::MaskCollection<eetest::CandForTest>",expr);

  auto func = parser.expr<MyExpr>();

  func->eval(c,r);

  std::cout << r.size()  << ' '  <<  std::count(r.begin(),r.end(),true) << std::endl;


  std::string cut = "bool operator()(int i, int j) const override { return i<10&& j<5; }";

  // ExpressionEvaluator parser2("ExpressionEvaluatorTests/EEUnitTest","eetest::vcut",cut.c_str());
  // auto mcut = parser2.expr<eetest::vcut>();

  auto const & mcut = *reco_expressionEvaluator("CommonTools/Utils",SINGLE_ARG(reco::genericExpression<bool, int, int>),cut);

  std::cout << mcut(2,7) << ' ' << mcut(3, 4) << std::endl;

  try {
    std::string cut = "bool operator()(int i, int j) const override { return i<10&& j<5; }";
    ExpressionEvaluator parser2("Bla/Blo","eetest::vcut",cut);
    auto mcut = parser2.expr<vcut>();
    std::cout << (*mcut)(2,7) << ' ' << (*mcut)(3, 4) << std::endl;
  }catch( cms::Exception const & e) {
    std::cout << e.what()  << std::endl;
  }catch(...) {
    std::cout << "unknown error...." << std::endl;
  }


  try {
    std::string cut = "bool operator()(int i, int j) override { return i<10&& j<5; }";
    auto const & mcut = *reco_expressionEvaluator("CommonTools/Utils",SINGLE_ARG(reco::genericExpression<bool,int,int>),cut);
    std::cout << mcut(2,7) << ' ' << mcut(3, 4) << std::endl;
 
 }catch( cms::Exception const & e) {
    std::cout << e.what()  << std::endl;
  }catch(...) {
    std::cout << "unknown error...." << std::endl;
  }


  // stress test
  std::atomic<int> j(0);
#pragma omp parallel num_threads(8)
  {
    reco::genericExpression<bool, int, int> const * acut = nullptr;
    for (int i=0; i<200; ++i) {
      acut = reco_expressionEvaluator("CommonTools/Utils",SINGLE_ARG(reco::genericExpression<bool, int, int>),cut);
      (*acut)(2,7);
      std::cerr << j++ <<',';
    }
  }
  std::cerr << std::endl;



  std::cout << "If HERE OK" << std::endl;

  return 0;

}
