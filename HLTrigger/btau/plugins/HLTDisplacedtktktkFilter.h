#ifndef HLTDisplacedtktktkFilter_h
#define HLTDisplacedtktktkFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
namespace edm {
  class ConfigurationDescriptions;
}
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

class HLTDisplacedtktktkFilter : public HLTFilter {
public:
  explicit HLTDisplacedtktktkFilter(const edm::ParameterSet&);
  ~HLTDisplacedtktktkFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginJob() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  void endJob() override;

private:
  bool fastAccept_;
  double minLxySignificance_;
  double maxLxySignificance_;
  double maxNormalisedChi2_;
  double minVtxProbability_;
  double minCosinePointingAngle_;
  const int triggerTypeDaughters_;

  edm::InputTag DisplacedVertexTag_;
  edm::EDGetTokenT<reco::VertexCollection> DisplacedVertexToken_;
  edm::InputTag beamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::InputTag TrackTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> TrackToken_;
};
#endif
