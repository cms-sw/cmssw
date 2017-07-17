#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonMatchCut : public CutApplicatorBase
{
public:
  MuonMatchCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }
  double value(const reco::CandidatePtr&) const override final;

private:
  const int minNumberOfMatchedStations_;

};
DEFINE_EDM_PLUGIN(CutApplicatorFactory, MuonMatchCut, "MuonMatchCut");

// Define constructors and initialization routines
MuonMatchCut::MuonMatchCut(const edm::ParameterSet& c):
  CutApplicatorBase(c),
  minNumberOfMatchedStations_(c.getParameter<int>("minNumberOfMatchedStations"))
{
}

// Functors for evaluation
CutApplicatorBase::result_type MuonMatchCut::operator()(const reco::MuonPtr& muon) const
{
  return muon->numberOfMatchedStations() >= minNumberOfMatchedStations_;
}

double MuonMatchCut::value(const reco::CandidatePtr& cand) const
{
  const reco::MuonPtr muon(cand);
  return muon->numberOfMatchedStations();
}
