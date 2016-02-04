#ifndef MYCLUSTER_H
#define MYCLUSTER_H
#include "CLHEP/Vector/LorentzVector.h"

enum {ClusterEm=0,ClusterHd=1,ClusterEmHd=2,ClusterTower=3,RecHitEm=4,RecHitHd=5,CaloTowerEm=6,CaloTowerHd=7};

struct MatchParam{
  int index;
  double distance;
};

struct CalCell{
  CLHEP::HepLorentzVector Momentum;
  int   pid;
  bool used;
};


struct CalCluster{
  CLHEP::HepLorentzVector Momentum;
  double em;
  double hd;
  int    type;
  int ncells;
  std::vector<CalCell> clusterCellList;
  std::vector<MatchParam> MatchedClusters;
  std::vector<CalCluster> SubClusterList;
};

class CellGreater {
  public:
  bool operator () (const CalCell& i, const CalCell& j) {
    return (i.Momentum.e() > j.Momentum.e());
  }
};

class CellEtGreater {
  public:
  bool operator () (const CalCell& i, const CalCell& j) {
    return (i.Momentum.perp() > j.Momentum.perp());
  }
};

class ClusterGreater {
  public:
  bool operator () (const CalCluster& i, const CalCluster& j) {
    return (i.Momentum.e() > j.Momentum.e());
  }
};

class ClusterEtGreater {
  public:
  bool operator () (const CalCluster& i, const CalCluster& j) {
    return (i.Momentum.perp() > j.Momentum.perp());
  }
};
class ClusterPtGreater {
  public:

  bool operator () (const CalCluster& i, const CalCluster& j) {
    return (i.Momentum.perp() > j.Momentum.perp());
  }
};


#endif
