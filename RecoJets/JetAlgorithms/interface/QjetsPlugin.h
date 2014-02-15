#ifndef _QJETSPLUGIN_
#define _QJETSPLUGIN_
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "RecoJets/JetAlgorithms/interface/Qjets.h"

class QjetsPlugin: public fastjet::JetDefinition::Plugin{
 private:
  bool _rand_seed_set;
  unsigned int _seed;
  int _truncated_length;
  double _zcut, _dcut_fctr, _exp_min, _exp_max, _rigidity,_truncation_fctr;
 public:
  QjetsPlugin(double zcut, double dcut_fctr, double exp_min, double exp_max, double rigidity, double truncation_fctr = 0.);
  void SetRandSeed(unsigned int seed); /* In case you want reproducible behavior */
  double R() const;
  string description() const;
  void run_clustering(fastjet::ClusterSequence & cs) const;
};
#endif
