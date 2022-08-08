#ifndef HLTMuonTrkL1TkMuFilter_h
#define HLTMuonTrkL1TkMuFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonTrkL1TkMuFilter : public HLTFilter {
public:
  HLTMuonTrkL1TkMuFilter(const edm::ParameterSet&);
  ~HLTMuonTrkL1TkMuFilter() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  // WARNING: two input collection represent should be aligned and represent
  // the same list of muons, just stored in different containers
  edm::InputTag m_muonsTag;                             // input collection of muons
  edm::EDGetTokenT<reco::MuonCollection> m_muonsToken;  // input collection of muons
  edm::InputTag m_candsTag;                             // input collection of candidates to be referenced
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> m_candsToken;  // input collection of candidates to be referenced
  edm::InputTag m_previousCandTag;  // input tag identifying product contains muons passing the previous level
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>
      m_previousCandToken;  // token identifying product contains muons passing the previous level
  int m_minTrkHits;
  int m_minMuonHits;
  int m_minMuonStations;
  double m_maxNormalizedChi2;
  double m_minPt;
  unsigned int m_minN;
  double m_maxAbsEta;
  bool m_saveTags;
};

#endif  //HLTMuonTrkL1TkMuFilter_h
