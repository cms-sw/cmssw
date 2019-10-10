#ifndef RecoHGCal_TICL_TracksterMomentumPluginBase_H__
#define RecoHGCal_TICL_TracksterMomentumPluginBase_H__

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ticl {
  class TracksterMomentumPluginBase {
  public:
    TracksterMomentumPluginBase(const edm::ParameterSet&, edm::ConsumesCollector&& iC) {}
    typedef reco::Candidate::LorentzVector LorentzVector;
    virtual ~TracksterMomentumPluginBase() {}
    virtual void setP4(const std::vector<const Trackster*>& tracksters,
                       std::vector<TICLCandidate>& ticl_cands,
                       edm::Event& event) const = 0;
  };
}  // namespace ticl

typedef edmplugin::PluginFactory<ticl::TracksterMomentumPluginBase*(const edm::ParameterSet&,
                                                                    edm::ConsumesCollector&& iC)>
    TracksterMomentumPluginFactory;

#endif
