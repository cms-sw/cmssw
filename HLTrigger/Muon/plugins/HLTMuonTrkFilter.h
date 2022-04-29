#ifndef HLTMuonTrkFilter_h
#define HLTMuonTrkFilter_h
// author D. Olivito
//  based on HLTDiMuonGlbTrkFilter.h
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonTrkFilter : public HLTFilter {
public:
  HLTMuonTrkFilter(const edm::ParameterSet&);
  ~HLTMuonTrkFilter() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  const PropagateToMuonSetup propSetup_;
  // WARNING: two input collection represent should be aligned and represent
  // the same list of muons, just stored in different containers
  const edm::InputTag m_muonsTag;                             // input collection of muons
  const edm::EDGetTokenT<reco::MuonCollection> m_muonsToken;  // input collection of muons
  const edm::InputTag m_candsTag;                             // input collection of candidates to be referenced
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection>
      m_candsToken;                       // input collection of candidates to be referenced
  const edm::InputTag m_previousCandTag;  // input tag identifying product contains muons passing the previous level
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>
      m_previousCandToken;  // token identifying product contains muons passing the previous level
  const int m_minTrkHits;
  const int m_minMuonHits;
  const int m_minMuonStations;
  const double m_maxNormalizedChi2;
  const unsigned int m_allowedTypeMask;
  const unsigned int m_requiredTypeMask;
  const muon::SelectionType m_trkMuonId;
  const double m_minPt;
  const unsigned int m_minN;
  const double m_maxAbsEta;
  const double m_l1MatchingdR;
  const double m_l1MatchingdR2;

  bool m_saveTags;
};

#endif  //HLTMuonTrkFilter_h
