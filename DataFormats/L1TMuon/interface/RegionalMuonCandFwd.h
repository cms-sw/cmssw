#ifndef __l1t_regional_muon_candidatefwd_h__
#define __l1t_regional_muon_candidatefwd_h__

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1TObjComparison.h"

namespace l1t {
  enum tftype { bmtf, omtf_neg, omtf_pos, emtf_neg, emtf_pos };
  class RegionalMuonCand;
  typedef BXVector<RegionalMuonCand> RegionalMuonCandBxCollection;

  typedef ObjectRef<RegionalMuonCand> RegionalMuonCandRef;
  typedef ObjectRefBxCollection<RegionalMuonCand> RegionalMuonCandRefBxCollection;
  typedef ObjectRefPair<RegionalMuonCand> RegionalMuonCandRefPair;
  typedef ObjectRefPairBxCollection<RegionalMuonCand> RegionalMuonCandRefPairBxCollection;
}  // namespace l1t

#endif /* define __l1t_regional_muon_candidatefwd_h__ */
