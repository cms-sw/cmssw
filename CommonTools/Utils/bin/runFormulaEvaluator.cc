#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {

  while(true) {
    std::cout <<">type in formula\n>"<<std::flush;
    
    std::string form;
    std::cin >> form;

    reco::FormulaEvaluator eval(form);

    std::vector<double> x;
    std::vector<double> v;

    std::cout << eval.evaluate(x,v)<<std::endl;
  }

  return 0;
}
