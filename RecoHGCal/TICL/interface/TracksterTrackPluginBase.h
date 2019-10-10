// Base class for plugins that set the track reference(s) in the Trackster -> TICLCandidate conversion.

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
    virtual void setTrack(const std::vector<const Trackster*>& tracksters,
                          std::vector<TICLCandidate>& ticl_cands,
                          edm::Event& event) const = 0;
  };
}  // namespace ticl

typedef edmplugin::PluginFactory<ticl::TracksterTrackPluginBase*(const edm::ParameterSet&, edm::ConsumesCollector&& iC)>
    TracksterTrackPluginFactory;

#endif
