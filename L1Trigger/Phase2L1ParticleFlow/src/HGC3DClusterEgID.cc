#include "L1Trigger/Phase2L1ParticleFlow/interface/HGC3DClusterEgID.h"
#include "CommonTools/MVAUtils/interface/TMVAZipReader.h"

l1tpf::HGC3DClusterEgID::HGC3DClusterEgID(const edm::ParameterSet &pset)
    : isPUFilter_(pset.getParameter<bool>("isPUFilter")),
      preselection_(pset.getParameter<std::string>("preselection")),
      method_(pset.getParameter<std::string>("method")),
      weightsFile_(pset.getParameter<std::string>("weightsFile")),
      reader_(new TMVA::Reader()),
      wp_(pset.getParameter<std::string>("wp")) {
  // first create all the variables
  for (const auto &psvar : pset.getParameter<std::vector<edm::ParameterSet>>("variables")) {
    variables_.emplace_back(psvar.getParameter<std::string>("name"), psvar.getParameter<std::string>("value"));
  }
}

void l1tpf::HGC3DClusterEgID::prepareTMVA() {
  // Declare the variables
  for (auto &var : variables_)
    var.declare(*reader_);
  // then read the weights
  if (weightsFile_[0] != '/' && weightsFile_[0] != '.') {
    weightsFile_ = edm::FileInPath(weightsFile_).fullPath();
  }
  reco::details::loadTMVAWeights(&*reader_, method_, weightsFile_);
}

bool l1tpf::HGC3DClusterEgID::passID(const l1t::HGCalMulticluster c, float &mvaOut) {
  if (preselection_(c)) {
    for (auto &var : variables_)
      var.fill(c);
    mvaOut = reader_->EvaluateMVA(method_);
    return mvaOut > wp_(c);
  } else {
    mvaOut = -100.;
    return false;
  }
}
