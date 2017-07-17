#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonPOGStandardCut : public CutApplicatorWithEventContentBase
{
public:
  MuonPOGStandardCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  double value(const reco::CandidatePtr& cand) const override final;

private:
  enum CutType {
    LOOSE, MEDIUM, TIGHT, SOFT, HIGHPT,
    NONE
  } cutType_;
  edm::Handle<reco::VertexCollection> vtxs_;
};
DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  MuonPOGStandardCut, "MuonPOGStandardCut");

// Define constructors and initialization routines
MuonPOGStandardCut::MuonPOGStandardCut(const edm::ParameterSet& c):
  CutApplicatorWithEventContentBase(c)
{
  const auto cutTypeName = c.getParameter<std::string>("idName");
  if ( cutTypeName == "loose") cutType_ = LOOSE;
  else if ( cutTypeName == "tight") cutType_ = TIGHT;
  else if ( cutTypeName == "medium") cutType_ = MEDIUM;
  else if ( cutTypeName == "soft" ) cutType_ = SOFT;
  else if ( cutTypeName == "highpt" ) cutType_ = HIGHPT;
  else
  {
    edm::LogError("MuonPOGStandardCut") << "Wrong cut id name, " << cutTypeName;
    cutType_ = NONE;
  }

  contentTags_.emplace("vertices", c.getParameter<edm::InputTag>("vertexSrc"));
}

void MuonPOGStandardCut::setConsumes(edm::ConsumesCollector& cc)
{
  auto vtcs = cc.consumes<reco::VertexCollection>(contentTags_["vertices"]);
  contentTokens_.emplace("vertices", vtcs);
}

void MuonPOGStandardCut::getEventContent(const edm::EventBase& ev)
{
  ev.getByLabel(contentTags_["vertices"], vtxs_);
}

// Functors for evaluation
CutApplicatorBase::result_type MuonPOGStandardCut::operator()(const reco::MuonPtr& cand) const
{
  switch( cutType_ ){
  case LOOSE:
    return muon::isLooseMuon(*cand);
    break;
  case TIGHT:
    return muon::isTightMuon(*cand, vtxs_->at(0));
    break;
  case MEDIUM:
    return muon::isMediumMuon(*cand);
    break;
  case SOFT:
    return muon::isSoftMuon(*cand, vtxs_->at(0));
    break;
  case HIGHPT:
    return muon::isHighPtMuon(*cand, vtxs_->at(0));
    break;
  case NONE:
    return false;
    break;
  }
  
  return true;
}

double MuonPOGStandardCut::value(const reco::CandidatePtr& cand) const {
  edm::Ptr<reco::Muon> mu(cand);
  switch( cutType_ ){
  case LOOSE:
    return muon::isLooseMuon(*mu);
    break;
  case TIGHT:
    return muon::isTightMuon(*mu, vtxs_->at(0));
    break;
  case MEDIUM:
    return muon::isMediumMuon(*mu);
    break;
  case SOFT:
    return muon::isSoftMuon(*mu, vtxs_->at(0));
    break;
  case HIGHPT:
    return muon::isHighPtMuon(*mu, vtxs_->at(0));
    break;
  case NONE:
    return 0.0;
    break;
  }
  return 1.0;
 }
