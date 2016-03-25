#ifndef HLTMuonTrkL1TFilter_h
#define HLTMuonTrkL1TFilter_h
// author D. Olivito
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTMuonTrkL1TFilter : public HLTFilter {
 public:
  HLTMuonTrkL1TFilter(const edm::ParameterSet&);
  virtual ~HLTMuonTrkL1TFilter(){}
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

 private:
  // WARNING: two input collection represent should be aligned and represent
  // the same list of muons, just stored in different containers
  edm::InputTag                                          m_muonsTag;   // input collection of muons
  edm::EDGetTokenT<reco::MuonCollection>                 m_muonsToken; // input collection of muons
  edm::InputTag                                          m_candsTag;   // input collection of candidates to be referenced
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> m_candsToken; // input collection of candidates to be referenced
  edm::InputTag                                          m_previousCandTag;   // input tag identifying product contains muons passing the previous level
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_previousCandToken; // token identifying product contains muons passing the previous level
  int m_minTrkHits;
  int m_minMuonHits;
  int m_minMuonStations;
  unsigned int m_allowedTypeMask;
  unsigned int m_requiredTypeMask;
  double m_maxNormalizedChi2;
  double m_minPt;
  unsigned int m_minN;
  double m_maxAbsEta;
  muon::SelectionType m_trkMuonId;
  bool m_saveTags;

};

#endif //HLTMuonTrkL1TFilter_h
