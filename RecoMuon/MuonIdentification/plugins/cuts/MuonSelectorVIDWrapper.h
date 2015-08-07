#ifndef __RecoMuon_MuonIdentification_MuonSelectorVIDWrapper_h__
#define __RecoMuon_MuonIdentification_MuonSelectorVIDWrapper_h__

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

template<muon::SelectionType selectionType,reco::Muon::ArbitrationType arbitrationType = reco::Muon::SegmentAndTrackArbitration>
  class MuonSelectorVIDWrapper : public CutApplicatorBase {
 public:
 MuonSelectorVIDWrapper(const edm::ParameterSet& c) :
 CutApplicatorBase(c) { }

 double value(const reco::CandidatePtr& cand) const override final {
   edm::Ptr<reco::Muon> mu(cand);
   return muon::isGoodMuon(*mu,selectionType,arbitrationType);
 }

 result_type operator()(const edm::Ptr<reco::Muon> & muon) const override final {
   return muon::isGoodMuon(*muon,selectionType,arbitrationType);
 }

 CandidateType candidateType() const override final {
   return MUON;
 }
};

#endif
