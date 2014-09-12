#ifndef RecoJets_JetAlgorithms_QJETSPLUGIN_h
#define RecoJets_JetAlgorithms_QJETSPLUGIN_h
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
  CLHEP::HepRandomEngine* _rnEngine;
 public:
  QjetsPlugin(double zcut, double dcut_fctr, double exp_min, double exp_max, double rigidity, double truncation_fctr = 0.)  : _rand_seed_set(false),
    _zcut(zcut),
    _dcut_fctr(dcut_fctr),
    _exp_min(exp_min),
    _exp_max(exp_max),
    _rigidity(rigidity),
    _truncation_fctr(truncation_fctr),
    _rnEngine(0)
      {};
  void SetRandSeed(unsigned int seed); /* In case you want reproducible behavior */
  void SetRNEngine(CLHEP::HepRandomEngine* rnEngine){
    _rnEngine=rnEngine;
  };
  double R() const;
  std::string description() const;
  void run_clustering(fastjet::ClusterSequence & cs) const;
};
#endif
