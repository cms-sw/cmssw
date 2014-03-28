#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/DeadChannelNNContext.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>

DeadChannelNNContext::DeadChannelNNContext() {
  for (int id = 0; id < NetworkID::lastID; ++id) {
    ctx[id].mlp = NULL;
  }
}

DeadChannelNNContext::~DeadChannelNNContext() {
  for (int id = 0; id < NetworkID::lastID; ++id) {
    if (ctx[id].mlp) {
      // @TODO segfaults for an uknown reason
      // delete ctx[id].mlp;
      // delete ctx[id].tree;
    }
  }
}

void DeadChannelNNContext::load_file(NetworkID id, std::string fn) {
  std::string path = edm::FileInPath(fn).fullPath();

  TTree *t = new TTree("t", "dummy MLP tree");
  t->SetDirectory(0);

  t->Branch("z1", &(ctx[id].tmp[0]), "z1/D");
  t->Branch("z2", &(ctx[id].tmp[1]), "z2/D");
  t->Branch("z3", &(ctx[id].tmp[2]), "z3/D");
  t->Branch("z4", &(ctx[id].tmp[3]), "z4/D");
  t->Branch("z5", &(ctx[id].tmp[4]), "z5/D");
  t->Branch("z6", &(ctx[id].tmp[5]), "z6/D");
  t->Branch("z7", &(ctx[id].tmp[6]), "z7/D");
  t->Branch("z8", &(ctx[id].tmp[7]), "z8/D");
  t->Branch("zf", &(ctx[id].tmp[8]), "zf/D");

  ctx[id].tree = t;
  ctx[id].mlp =
      new TMultiLayerPerceptron("@z1,@z2,@z3,@z4,@z5,@z6,@z7,@z8:10:5:zf", t);
  ctx[id].mlp->LoadWeights(path.c_str());
}

void DeadChannelNNContext::load() {
  std::string p = "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/NNWeights/";

  this->load_file(NetworkID::ccEB, p + "EB_ccNNWeights.txt");
  this->load_file(NetworkID::rrEB, p + "EB_rrNNWeights.txt");
  this->load_file(NetworkID::llEB, p + "EB_llNNWeights.txt");
  this->load_file(NetworkID::uuEB, p + "EB_uuNNWeights.txt");
  this->load_file(NetworkID::ddEB, p + "EB_ddNNWeights.txt");
  this->load_file(NetworkID::ruEB, p + "EB_ruNNWeights.txt");
  this->load_file(NetworkID::rdEB, p + "EB_rdNNWeights.txt");
  this->load_file(NetworkID::luEB, p + "EB_luNNWeights.txt");
  this->load_file(NetworkID::ldEB, p + "EB_ldNNWeights.txt");

  this->load_file(NetworkID::ccEE, p + "EE_ccNNWeights.txt");
  this->load_file(NetworkID::rrEE, p + "EE_rrNNWeights.txt");
  this->load_file(NetworkID::llEE, p + "EE_llNNWeights.txt");
  this->load_file(NetworkID::uuEE, p + "EE_uuNNWeights.txt");
  this->load_file(NetworkID::ddEE, p + "EE_ddNNWeights.txt");
  this->load_file(NetworkID::ruEE, p + "EE_ruNNWeights.txt");
  this->load_file(NetworkID::rdEE, p + "EE_rdNNWeights.txt");
  this->load_file(NetworkID::luEE, p + "EE_luNNWeights.txt");
  this->load_file(NetworkID::ldEE, p + "EE_ldNNWeights.txt");
}

double DeadChannelNNContext::value(NetworkID method, int index, double in0,
                                   double in1, double in2, double in3,
                                   double in4, double in5, double in6,
                                   double in7) {

  if (!ctx[method].mlp) this->load();

  double vCC[8] = { in0, in1, in2, in3, in4, in5, in6, in7 };
  return ctx[method].mlp->Evaluate(0, vCC);
}
