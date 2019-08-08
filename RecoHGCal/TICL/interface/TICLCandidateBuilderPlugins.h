#ifndef RecoHGCal_TICL_TICLCandidateBuilderPlugins_H__
#define RecoHGCal_TICL_TICLCandidateBuilderPlugins_H__

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ticl {
  class TracksterMomentumBase {
  public:
    typedef reco::Candidate::LorentzVector LorentzVector;
    virtual ~TracksterMomentumBase(){};
    virtual LorentzVector calcP4(const ticl::Trackster& trackster, const reco::Vertex& vertex) const = 0;
  };
} // namespace

typedef edmplugin::PluginFactory<ticl::TracksterMomentumBase*()> TracksterMomentumPluginFactory;

#endif
