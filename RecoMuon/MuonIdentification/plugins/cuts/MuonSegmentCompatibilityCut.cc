#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonSegmentCompatibilityCut : public CutApplicatorBase
{
public:
  MuonSegmentCompatibilityCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }
  double value(const reco::CandidatePtr&) const override final;

private:
  double maxGlbNormChi2_, maxChi2LocalPos_, maxTrkKink_;
  const double minCompatGlb_, minCompatNonGlb_;

};
DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  MuonSegmentCompatibilityCut, "MuonSegmentCompatibilityCut");

// Define constructors and initialization routines
MuonSegmentCompatibilityCut::MuonSegmentCompatibilityCut(const edm::ParameterSet& c):
  CutApplicatorBase(c),
  minCompatGlb_(c.getParameter<double>("minCompatGlb")),
  minCompatNonGlb_(c.getParameter<double>("minCompatNonGlb"))
{
  const edm::ParameterSet cc = c.getParameter<edm::ParameterSet>("goodGLB");
  maxGlbNormChi2_ = cc.getParameter<double>("maxGlbNormChi2");
  maxChi2LocalPos_ = cc.getParameter<double>("maxChi2LocalPos");
  maxTrkKink_ = cc.getParameter<double>("maxTrkKink");
}

// Functors for evaluation
CutApplicatorBase::result_type MuonSegmentCompatibilityCut::operator()(const reco::MuonPtr& muon) const
{
  const bool isGoodGlb = (
    muon->isGlobalMuon() and
    muon->globalTrack()->normalizedChi2() < maxGlbNormChi2_ and
    muon->combinedQuality().chi2LocalPosition < maxChi2LocalPos_ and
    muon->combinedQuality().trkKink < maxTrkKink_
  );

  const double compat = muon::segmentCompatibility(*muon);

  return compat > (isGoodGlb ? minCompatGlb_ : minCompatNonGlb_);

}

double MuonSegmentCompatibilityCut::value(const reco::CandidatePtr& cand) const
{
  const reco::MuonPtr muon(cand);
  return muon::segmentCompatibility(*muon);
}
