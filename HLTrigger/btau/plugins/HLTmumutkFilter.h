#ifndef HLTmumutkFilter_h
#define HLTmumutkFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
namespace edm {
  class ConfigurationDescriptions;
}
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
// ----------------------------------------------------------------------

class HLTmumutkFilter : public HLTFilter {

 public:
  explicit HLTmumutkFilter(const edm::ParameterSet&);
  ~HLTmumutkFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

 private:

  edm::InputTag                                          muCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muCandToken_;
  edm::InputTag                                          trkCandTag_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> trkCandToken_;
  edm::InputTag                                          MuMuTkVertexTag_;
  edm::EDGetTokenT<reco::VertexCollection>               MuMuTkVertexToken_;
  edm::InputTag                            				 beamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot>         				 beamSpotToken_;

  const double maxEta_;
  const double minPt_;
  const double maxNormalisedChi2_;
  const double minVtxProbability_;
  const double minLxySignificance_;
  const double minCosinePointingAngle_;

  static bool triggerdByPreviousLevel(const reco::RecoChargedCandidateRef &, const std::vector<reco::RecoChargedCandidateRef> &);

};
#endif
