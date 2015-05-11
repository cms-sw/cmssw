#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class RecoMuonBaseIDCut : public CutApplicatorWithEventContentBase
{
public:
  RecoMuonBaseIDCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

private:
  enum CutType {
    LOOSE, MEDIUM, TIGHT, SOFT, HIGHPT,
    NONE
  } cutType_;
  edm::Handle<reco::VertexCollection> vtxs_;
};
DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  RecoMuonBaseIDCut, "RecoMuonBaseIDCut");

// Define constructors and initialization routines
RecoMuonBaseIDCut::RecoMuonBaseIDCut(const edm::ParameterSet& c):
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
    edm::LogError("RecoMuonBaseIDCut") << "Wrong cut id name, " << cutTypeName;
    cutType_ = NONE;
  }

  contentTags_.emplace("vertices", c.getParameter<edm::InputTag>("vertexSrc"));
}

void RecoMuonBaseIDCut::setConsumes(edm::ConsumesCollector& cc)
{
  auto vtcs = cc.consumes<reco::VertexCollection>(contentTags_["vertices"]);
  contentTokens_.emplace("vertices", vtcs);
}

void RecoMuonBaseIDCut::getEventContent(const edm::EventBase& ev)
{
  ev.getByLabel(contentTags_["vertices"], vtxs_);
}

// Functors for evaluation
CutApplicatorBase::result_type RecoMuonBaseIDCut::operator()(const reco::MuonPtr& cand) const
{
  if ( cutType_ == LOOSE ) return muon::isLooseMuon(*cand);
  else if ( cutType_ == TIGHT ) return muon::isTightMuon(*cand, vtxs_->at(0));
  else if ( cutType_ == MEDIUM ) return muon::isMediumMuon(*cand);
  else if ( cutType_ == SOFT ) return muon::isSoftMuon(*cand, vtxs_->at(0));
  else if ( cutType_ == HIGHPT ) return muon::isHighPtMuon(*cand, vtxs_->at(0));

  return true;
}


