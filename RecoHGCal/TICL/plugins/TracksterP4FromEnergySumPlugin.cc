#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoHGCal/TICL/interface/TICLCandidateBuilderPlugins.h"

namespace ticl {
  class TracksterP4FromEnergySum : public TracksterMomentumBase {
  public:
    virtual LorentzVector calcP4(const ticl::Trackster& trackster, const reco::Vertex& vertex) const final;
  };

  TracksterP4FromEnergySum::LorentzVector TracksterP4FromEnergySum::calcP4(const ticl::Trackster& trackster, const reco::Vertex& vertex) const {
    TracksterP4FromEnergySum::LorentzVector p4;
    return p4;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(TracksterMomentumPluginFactory,
                  ticl::TracksterP4FromEnergySum,
                  "TracksterP4FromEnergySum");
