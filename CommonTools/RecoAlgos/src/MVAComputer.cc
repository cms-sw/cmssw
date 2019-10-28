#include "CommonTools/RecoAlgos/interface/MVAComputer.h"

MVAComputer::MVAComputer(mva_variables* vars, std::string weights_file) {
  vars_ = vars;
  reader_ = std::make_unique<TMVA::Reader>("!Color:Silent");
  for (auto& var : *vars_)
    reader_->AddVariable(std::get<0>(var), &std::get<1>(var));

  reader_->BookMVA("BDT", weights_file);
}

float MVAComputer::operator()() { return 1. / (1 + sqrt(2 / (1 + reader_->EvaluateMVA("BDT")) - 1)); }

MVAComputer& MVAComputer::operator=(MVAComputer&& other) {
  if (this != &other) {
    reader_ = std::exchange(other.reader_, nullptr);
    vars_ = std::exchange(other.vars_, nullptr);
  }

  return *this;
};
