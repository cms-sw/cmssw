#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonMomQualityCut : public CutApplicatorBase
{
public:
  MuonMomQualityCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }
  double value(const reco::CandidatePtr&) const override final;

private:
  const double maxRelPtErr_;
};
DEFINE_EDM_PLUGIN(CutApplicatorFactory, MuonMomQualityCut, "MuonMomQualityCut");

// Define constructors and initialization routines
MuonMomQualityCut::MuonMomQualityCut(const edm::ParameterSet& c):
  CutApplicatorBase(c),
  maxRelPtErr_(c.getParameter<double>("maxRelPtErr"))
{
}

// Functors for evaluation
CutApplicatorBase::result_type MuonMomQualityCut::operator()(const reco::MuonPtr& cand) const
{
  const auto trackRef = cand->muonBestTrack();
  return trackRef.isNonnull() and trackRef->ptError() <= maxRelPtErr_*trackRef->pt();

  return true;
}

double MuonMomQualityCut::value(const reco::CandidatePtr& cand) const
{
  const reco::MuonPtr muon(cand);
  const auto trackRef = muon->muonBestTrack();
  if ( trackRef.isNull() or trackRef->pt() <= 0 ) return -1;

  return trackRef->ptError()/trackRef->pt();
}
