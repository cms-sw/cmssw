#include "../interface/GenerateOnly.h"
#include "../interface/Combine.h"
#include <iostream>

using namespace RooStats;

GenerateOnly::GenerateOnly() :
    LimitAlgo("GenerateOnly specific options: none") {
}

void GenerateOnly::applyOptions(const boost::program_options::variables_map &vm) {
}

bool GenerateOnly::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
  if (verbose > 0) {
    std::cout << "generate toy samples only; no limit computation " << std::endl;
  }
  return true;
}
