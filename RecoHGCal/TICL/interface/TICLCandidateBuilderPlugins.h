#ifndef RecoHGCal_TICL_TICLCandidateBuilderPlugins_H__
#define RecoHGCal_TICL_TICLCandidateBuilderPlugins_H__

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ticl {
  class TracksterMomentumPluginBase {
  public:
    // TracksterMomentumPluginBase() {}
    TracksterMomentumPluginBase(const edm::ParameterSet&, edm::ConsumesCollector&& iC) {}
    typedef reco::Candidate::LorentzVector LorentzVector;
    virtual ~TracksterMomentumPluginBase(){};
    virtual LorentzVector calcP4(const ticl::Trackster& trackster) const = 0;

    edm::Event& evt() { return *evt_; }
    
    // To be 
    void beginEvent(edm::Event& evt) {
       evt_ = &evt;
       this->beginEvt();
     }
  private:
    virtual void beginEvt() = 0;
    edm::Event* evt_;
  };
} // namespace

typedef edmplugin::PluginFactory<ticl::TracksterMomentumPluginBase*(const edm::ParameterSet&, edm::ConsumesCollector&& iC)> TracksterMomentumPluginFactory;

#endif
