#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class MuonDzCut : public CutApplicatorWithEventContentBase
{
public:
  MuonDzCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const final;
  CandidateType candidateType() const final { return MUON; }
  double value(const reco::CandidatePtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

private:
  edm::Handle<reco::VertexCollection> vtxs_;
  const double maxDz_;
  enum BestTrackType { INNERTRACK, MUONBESTTRACK, NONE } trackType_;
};
DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  MuonDzCut, "MuonDzCut");

// Define constructors and initialization routines
MuonDzCut::MuonDzCut(const edm::ParameterSet& c):
  CutApplicatorWithEventContentBase(c),
  maxDz_(c.getParameter<double>("maxDz"))
{
  const std::string trackTypeName = c.getParameter<std::string>("trackType");
  trackType_ = NONE;
  if      ( trackTypeName == "muonBestTrack" ) trackType_ = MUONBESTTRACK;
  else if ( trackTypeName == "innerTrack" ) trackType_ = INNERTRACK;
  else
  {
    edm::LogError("MuonDzCut") << "Wrong cut id name, " << trackTypeName
                                        << "Choose among \"muonBestTrack\", \"innerTrack\"";
    trackType_ = NONE;
  }

  contentTags_.emplace("vertices", c.getParameter<edm::InputTag>("vertexSrc"));
  contentTags_.emplace("verticesMiniAOD", c.getParameter<edm::InputTag>("vertexSrcMiniAOD"));
}

void MuonDzCut::setConsumes(edm::ConsumesCollector& cc)
{
  contentTokens_.emplace("vertices", cc.consumes<reco::VertexCollection>(contentTags_["vertices"]));
  contentTokens_.emplace("verticesMiniAOD", cc.consumes<reco::VertexCollection>(contentTags_["verticesMiniAOD"]));
}

void MuonDzCut::getEventContent(const edm::EventBase& ev)
{
  ev.getByLabel(contentTags_["vertices"], vtxs_);
  if ( !vtxs_.isValid() ) ev.getByLabel(contentTags_["verticesMiniAOD"], vtxs_);
}

// Functors for evaluation
CutApplicatorBase::result_type MuonDzCut::operator()(const reco::MuonPtr& cand) const
{
  const auto& vtxPos = vtxs_->at(0).position();

  reco::TrackRef trackRef;
  if      ( trackType_ == INNERTRACK    ) trackRef = cand->innerTrack();
  else if ( trackType_ == MUONBESTTRACK ) trackRef = cand->muonBestTrack();

  return trackRef.isNonnull() and std::abs(trackRef->dz(vtxPos)) <= maxDz_;
}

double MuonDzCut::value(const reco::CandidatePtr& cand) const
{
  const reco::MuonPtr muon(cand);
  reco::TrackRef trackRef;
  if      ( trackType_ == INNERTRACK    ) trackRef = muon->innerTrack();
  else if ( trackType_ == MUONBESTTRACK ) trackRef = muon->muonBestTrack();
  if ( trackRef.isNull() ) return -1;

  const auto& vtxPos = vtxs_->at(0).position();
  return std::abs(trackRef->dz(vtxPos));
}
