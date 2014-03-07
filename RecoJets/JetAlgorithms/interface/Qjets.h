#ifndef RecoJets_JetAlgorithms_QJets_h
#define RecoJets_JetAlgorithms_QJets_h
#include <queue>
#include <vector>
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandomEngine.h"

struct JetDistance{
  double dij;
  int j1;
  int j2;
};

class JetDistanceCompare{
 public:
  JetDistanceCompare(){};
  bool operator() (const JetDistance& lhs, const JetDistance&rhs) const {return lhs.dij > rhs.dij;};
};

class Qjets{
 private:
  bool _rand_seed_set;
  unsigned int _seed;
  double _zcut, _dcut, _dcut_fctr, _exp_min, _exp_max, _rigidity, _truncation_fctr;
  std::map<int,bool> _merged_jets;
  std::priority_queue <JetDistance, std::vector<JetDistance>, JetDistanceCompare> _distances;
  CLHEP::HepRandomEngine* _rnEngine;

  double d_ij(const fastjet::PseudoJet& v1, const fastjet::PseudoJet& v2) const; 
  void computeDCut(fastjet::ClusterSequence & cs);

  double Rand();
  bool Prune(JetDistance& jd,fastjet::ClusterSequence & cs);
  bool JetsUnmerged(const JetDistance& jd) const;
  bool JetUnmerged(int num) const;
  void ComputeNewDistanceMeasures(fastjet::ClusterSequence & cs, unsigned int new_jet);
  void ComputeAllDistances(const std::vector<fastjet::PseudoJet>& inp);  
  double ComputeMinimumDistance();
  double ComputeNormalization(double dmin);
  JetDistance GetNextDistance();
  bool Same(const JetDistance& lhs, const JetDistance& rhs);

 public:
  Qjets(double zcut, double dcut_fctr, double exp_min, double exp_max, double rigidity, double truncation_fctr, CLHEP::HepRandomEngine* rnEngine)  : _rand_seed_set(false),
    _zcut(zcut),
    _dcut(-1.),
    _dcut_fctr(dcut_fctr),
    _exp_min(exp_min),
    _exp_max(exp_max),
    _rigidity(rigidity),
    _truncation_fctr(truncation_fctr),
    _rnEngine(rnEngine)
    {};
    
  void Cluster(fastjet::ClusterSequence & cs);
  void SetRandSeed(unsigned int seed); /* In case you want reproducible behavior */
};
#endif
