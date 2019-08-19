#ifndef RecoHGCal_TICL_TracksterTrackPluginBase_H__
#define RecoHGCal_TICL_TracksterTrackPluginBase_H__

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ticl {
  class TracksterTrackPluginBase {
  public:
    TracksterTrackPluginBase(const edm::ParameterSet&, edm::ConsumesCollector&& iC) {}
    typedef reco::Candidate::LorentzVector LorentzVector;
    virtual ~TracksterTrackPluginBase() {}
    virtual void setTrack(const ticl::Trackster& trackster, ticl::TICLCandidate& ticl_cand) const = 0;

    // Allow access to event so data products can be read once at beginning of event but four-vector calculation can 
    // be run on single tracksters
    edm::Event& evt() const { return *evt_; }
    
    // Needs to be called by CMSSW plugins using this plugin before being able to use other methods
    void beginEvent(edm::Event& evt) {
       evt_ = &evt;
       this->beginEvt();
     }
  private:
    virtual void beginEvt() = 0;
    edm::Event* evt_;
  };
} // namespace

typedef edmplugin::PluginFactory<ticl::TracksterTrackPluginBase*(const edm::ParameterSet&, edm::ConsumesCollector&& iC)> TracksterTrackPluginFactory;

#endif
