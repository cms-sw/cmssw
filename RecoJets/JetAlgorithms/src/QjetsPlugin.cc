#include "RecoJets/JetAlgorithms/interface/QjetsPlugin.h"

QjetsPlugin::QjetsPlugin(double zcut, double dcut_fctr, double exp_min, double exp_max, double rigidity)
  : _zcut(zcut), _dcut_fctr(dcut_fctr), _exp_min(exp_min), _exp_max(exp_max), _rigidity(rigidity), _rand_seed_set(false)
{
}

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
  Qjets qjets(_zcut, _dcut_fctr, _exp_min, _exp_max, _rigidity);
  if(_rand_seed_set)
    qjets.SetRandSeed(_seed);
  
  qjets.Cluster(cs);
}
