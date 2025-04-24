#include "L1Trigger/Phase2L1ParticleFlow/interface/HGC3DClusterID.h"

l1tpf::HGC3DClusterID::HGC3DClusterID(const edm::ParameterSet &pset) {
  // Inference of the conifer BDT model
  multiclass_bdt_ = new conifer::BDT<bdt_feature_t, bdt_score_t, false>(edm::FileInPath(pset.getParameter<std::string>("model")).fullPath());

  wp_PU = {0.38534376, 0.33586645};
  wp_Pi = {0.22037095, 0.08385937};
  wp_Eg = {0.13564333, 0.36078927};
}

float l1tpf::HGC3DClusterID::evaluate(const l1t::HGCalMulticluster &cl, l1t::PFCluster &cpf) {
  // Input for the BDT: showerlength, coreshowerlength, eot, eta, meanz, seetot, spptot, szz
  bdt_feature_t showerlength = cl.showerLength();
  bdt_feature_t coreshowerlength = cl.coreShowerLength();
  bdt_feature_t eot = cl.eot();
  bdt_feature_t eta = std::abs(cl.eta()); // take absolute values for eta for BDT input
  bdt_feature_t meanz = std::abs(cl.zBarycenter()) - 320;
  bdt_feature_t seetot = cl.sigmaEtaEtaTot();
  bdt_feature_t spptot = cl.sigmaPhiPhiTot();
  bdt_feature_t szz = cl.sigmaZZ();

  // Run BDT inference
  inputs = {showerlength, coreshowerlength, eot, eta, meanz, seetot, spptot, szz};

  bdt_score = multiclass_bdt_->decision_function(inputs);

  // BDT score
  float puRawScore = bdt_score[0];
  float emRawScore = bdt_score[2];
  float piRawScore = bdt_score[1];

  // softmax (for now, let's compute the softmax in this code; this needs to be changed to implement on firmware)
  // Softmax implemented in conifer (standalone) is to be integrated here soon; for now, just do "offline" softmax :(
  float denom = exp(puRawScore) + exp(emRawScore) + exp(piRawScore);
  float puScore = exp(puRawScore) / denom;
  float emScore = exp(emRawScore) / denom;
  float piScore = exp(piRawScore) / denom;

  // max score to ID the cluster -> Deprecated
  float maxScore = *std::max_element(bdt_score.begin(), bdt_score.end());

  cpf.setPuIDScore(puScore);
  cpf.setEmIDScore(emScore);
  cpf.setPiIDScore(piScore);
  
  cpf.setRho(cl.pt());

  return maxScore;
}


bool l1tpf::HGC3DClusterID::passPuID(l1t::PFCluster &cpf, float maxScore) {

  return (cpf.getRho() < 20 ? (cpf.puIDScore() > wp_PU[0]) : (cpf.puIDScore() > wp_PU[1]));
}

bool l1tpf::HGC3DClusterID::passPFEmID(l1t::PFCluster &cpf, float maxScore) {

  return (cpf.getRho() < 20 ? ((cpf.puIDScore() <= wp_PU[0]) && (cpf.emIDScore() > wp_Eg[0])) : ((cpf.puIDScore() <= wp_PU[1]) && (cpf.emIDScore() > wp_Eg[1])));
}

bool l1tpf::HGC3DClusterID::passEgEmID(l1t::PFCluster &cpf, float maxScore) {
 
  return (cpf.getRho() < 20 ? ((cpf.puIDScore() <= wp_PU[0]) && (cpf.emIDScore() > wp_Eg[0])) : ((cpf.puIDScore() <= wp_PU[1]) && (cpf.emIDScore() > wp_Eg[1])));
}


bool l1tpf::HGC3DClusterID::passPiID(l1t::PFCluster &cpf, float maxScore) {
  
  return (cpf.getRho() < 20 ? ((cpf.puIDScore() <= wp_PU[0]) && (cpf.piIDScore() > wp_Pi[0])) : ((cpf.puIDScore() <= wp_PU[1]) && (cpf.piIDScore() > wp_Pi[1])));
}
