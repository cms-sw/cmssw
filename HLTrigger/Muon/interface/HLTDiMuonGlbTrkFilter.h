#ifndef HLTDiMuonGlbTrkFilter_h
#define HLTDiMuonGlbTrkFilter_h
// author D.Kovalskyi
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTDiMuonGlbTrkFilter : public HLTFilter {
 public:
  HLTDiMuonGlbTrkFilter(const edm::ParameterSet&);
  virtual ~HLTDiMuonGlbTrkFilter(){}
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  // WARNING: two input collection represent should be aligned and represent
  // the same list of muons, just stored in different containers
  edm::InputTag m_muons;  // input collection of muons
  edm::InputTag m_cands;  // input collection of candidates to be referenced
  int m_minTrkHits;
  int m_minMuonHits;
  unsigned int m_allowedTypeMask;
  unsigned int m_requiredTypeMask;
  double m_maxNormalizedChi2;
  double m_minDR;
  double m_minPtMuon1;
  double m_minPtMuon2;
  double m_minMass;
  int m_minMatches;
  muon::SelectionType m_trkMuonId;
  bool m_saveTags;

};

#endif //HLTMuonDimuonFilter_h
