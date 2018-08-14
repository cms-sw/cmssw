#ifndef HLTmumutktkFilter_h
#define HLTmumutktkFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
namespace edm {
  class ConfigurationDescriptions;
}

// ----------------------------------------------------------------------

class HLTmumutktkFilter : public HLTFilter {

 public:
  explicit HLTmumutktkFilter(const edm::ParameterSet&);
  ~HLTmumutktkFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

 private:

  edm::InputTag                                          muCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muCandToken_;
  edm::InputTag                                          trkCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> trkCandToken_;
  edm::InputTag                                          MuMuTkVertexTag_;
  edm::EDGetTokenT<reco::VertexCollection>               MuMuTkVertexToken_;
  edm::InputTag                                          beamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot>                       beamSpotToken_;

  const double maxEta_;
  const double minPt_;
  const double maxNormalisedChi2_;
  const double minVtxProbability_;
  const double minLxySignificance_;
  const double minCosinePointingAngle_;

  static bool triggerdByPreviousLevel(const reco::RecoChargedCandidateRef &, const std::vector<reco::RecoChargedCandidateRef> &);

};
#endif
