#include "RecoJets/JetAlgorithms/interface/QjetsPlugin.h"

using namespace std;

void QjetsPlugin::SetRandSeed(unsigned int seed){
  _rand_seed_set = true;
  _seed = seed;
}

double QjetsPlugin::R()const{
  return 0.;
}

string QjetsPlugin::description() const{
  string desc("Qjets pruning plugin");
  return desc;
}

void QjetsPlugin::run_clustering(fastjet::ClusterSequence & cs) const{
  Qjets qjets(_zcut, _dcut_fctr, _exp_min, _exp_max, _rigidity, _truncation_fctr, _rnEngine);
  if(_rand_seed_set)
    qjets.SetRandSeed(_seed);
  qjets.Cluster(cs);
}
