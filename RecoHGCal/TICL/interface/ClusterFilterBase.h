// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

#ifndef RecoHGCal_TICL_ClusterFilterBase_H__
#define RecoHGCal_TICL_ClusterFilterBase_H__

#include <vector>
#include <memory>

namespace edm {class ParameterSet; class Event; class EventSetup;}


class ClusterFilterBase {
public:
  virtual ~ClusterFilterBase(){}

  virtual std::unique_ptr<std::vector<std::pair<unsigned int, float> > >  filter(const std::vector<reco::CaloCluster>& layerClusters,
                                           const std::vector<std::pair<unsigned int, float> >& mask) const { return std::unique_ptr<std::vector<std::pair<unsigned int, float> > >(); }

  virtual std::unique_ptr<std::vector<std::pair<unsigned int, float> > >  filter(
      const edm::EventSetup& es,
      const std::vector<reco::CaloCluster>& layerClusters,
      const std::vector<std::pair<unsigned int, float> >& mask) const { return nullptr;}

  virtual std::unique_ptr<std::vector<std::pair<unsigned int, float> > >  filter(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const std::vector<reco::CaloCluster>& layerClusters,
      const std::vector<std::pair<unsigned int, float> >& mask) const { return filter(es, layerClusters, mask); }
};


#endif
