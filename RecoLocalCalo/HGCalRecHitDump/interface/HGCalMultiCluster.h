#ifndef RecoHGCAL_HGCALClusters_HGCalMultiCluster
#define RecoHGCAL_HGCALClusters_HGCalMultiCluster

#include <vector>

#include "DataFormats/EgammaReco/interface/BasicCluster.h"


class HGCalMultiCluster{

public:
  typedef std::vector<reco::BasicCluster>::const_iterator component_iterator;
  typedef std::vector<reco::BasicCluster> ClusterCollection;

  HGCalMultiCluster(){
  }
  HGCalMultiCluster(ClusterCollection &thecls) : myclusters(thecls){
  }
  HGCalMultiCluster(const reco::BasicCluster &thecl) :  myclusters(1,thecl){
  }
  void push_back(const reco::BasicCluster &b){
    myclusters.push_back(b);
  }
  unsigned int size()
  {
    return myclusters.size();
  }						

  component_iterator begin(){
    return myclusters.begin();
  }
  component_iterator end(){
    return myclusters.end();
  }

private:

  std::vector<reco::BasicCluster> myclusters;

};
#endif
