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

float l1tpf::HGC3DClusterEgID::passID(l1t::HGCalMulticluster c, l1t::PFCluster &cpf) {
  if (preselection_(c)) {
    for (auto &var : variables_)
      var.fill(c);
    float mvaOut = reader_->EvaluateMVA(method_);
    if (isPUFilter_)
      cpf.setEgVsPUMVAOut(mvaOut);
    else
      cpf.setEgVsPionMVAOut(mvaOut);
    return (mvaOut > wp_(c) ? 1 : 0);
  } else {
    if (isPUFilter_)
      cpf.setEgVsPUMVAOut(-100.0);
    else
      cpf.setEgVsPionMVAOut(-100.0);
    return 0;
  }
}
