#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonMomQualityCut : public CutApplicatorBase
{
public:
  MuonMomQualityCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }

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
  if ( trackRef.isNull() ) return false;
  if ( trackRef->ptError()/trackRef->pt() > maxRelPtErr_ ) return false;

  return true;
}

