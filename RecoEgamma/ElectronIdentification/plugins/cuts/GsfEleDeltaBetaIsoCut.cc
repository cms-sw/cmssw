#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"


class GsfEleDeltaBetaIsoCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleDeltaBetaIsoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const float _isoCutEBLowPt,_isoCutEBHighPt,_isoCutEELowPt,_isoCutEEHighPt;
  const float _deltaBetaConstant,_ptCutOff,_barrelCutOff;
  bool _relativeIso;
  edm::Handle<edm::ValueMap<float> > _chad_iso,_nhad_iso,_ph_iso,_PUchad_iso;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDeltaBetaIsoCut,
		  "GsfEleDeltaBetaIsoCut");

GsfEleDeltaBetaIsoCut::GsfEleDeltaBetaIsoCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _isoCutEBLowPt(c.getParameter<double>("isoCutEBLowPt")),
  _isoCutEBHighPt(c.getParameter<double>("isoCutEBHighPt")),
  _isoCutEELowPt(c.getParameter<double>("isoCutEELowPt")),
  _isoCutEEHighPt(c.getParameter<double>("isoCutEEHighPt")),
  _deltaBetaConstant(c.getParameter<double>("deltaBetaConstant")),
  _ptCutOff(c.getParameter<double>("ptCutOff")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")),
  _relativeIso(c.getParameter<bool>("isRelativeIso")) {  
  edm::InputTag chadtag = c.getParameter<edm::InputTag>("chargedHadronIsoDR03Src");
  edm::InputTag nhadtag = c.getParameter<edm::InputTag>("neutralHadronIsoDR03Src");
  edm::InputTag phtag = c.getParameter<edm::InputTag>("photonIsoDR03Src");
  edm::InputTag PUchadtag = c.getParameter<edm::InputTag>("puChargedHadronIsoDR03Src");
  contentTags_.emplace("chad",chadtag);
  contentTags_.emplace("nhad",nhadtag);
  contentTags_.emplace("pho",phtag);
  contentTags_.emplace("puchad",PUchadtag);
}

void GsfEleDeltaBetaIsoCut::setConsumes(edm::ConsumesCollector& cc) {
  auto chad = 
    cc.consumes<edm::ValueMap<float> >(contentTags_["chad"]);
  contentTokens_.emplace("chad",chad);
  auto nhad = 
    cc.consumes<edm::ValueMap<float> >(contentTags_["nhad"]);
  contentTokens_.emplace("nhad",nhad);
  auto pho = 
    cc.consumes<edm::ValueMap<float> >(contentTags_["pho"]);
  contentTokens_.emplace("pho",pho);
  auto puchad = 
    cc.consumes<edm::ValueMap<float> >(contentTags_["puchad"]);
  contentTokens_.emplace("puchad",puchad);
}

void GsfEleDeltaBetaIsoCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_["chad"],_chad_iso);
  ev.getByLabel(contentTags_["nhad"],_nhad_iso);
  ev.getByLabel(contentTags_["pho"],_ph_iso);
  ev.getByLabel(contentTags_["puchad"],_PUchad_iso);
}

CutApplicatorBase::result_type 
GsfEleDeltaBetaIsoCut::
operator()(const reco::GsfElectronPtr& cand) const{ 
  const float isoCut = 
    ( cand->p4().pt() < _ptCutOff ? 
      ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ?
	_isoCutEBLowPt : _isoCutEELowPt ) :
      ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ?
	_isoCutEBHighPt : _isoCutEEHighPt ) );
  const float chad = (*_chad_iso)[cand];
  const float nhad = (*_nhad_iso)[cand];
  const float pho = (*_ph_iso)[cand];
  const float puchad = (*_PUchad_iso)[cand];
  float iso = chad + std::max(0.0f, nhad + pho - _deltaBetaConstant*puchad);
  if( _relativeIso ) iso /= cand->p4().pt();
  return iso < isoCut;
}

double GsfEleDeltaBetaIsoCut::value(const reco::CandidatePtr& cand) const {
  const float chad = (*_chad_iso)[cand];
  const float nhad = (*_nhad_iso)[cand];
  const float pho = (*_ph_iso)[cand];
  const float puchad = (*_PUchad_iso)[cand];
  float iso = chad + std::max(0.0f, nhad + pho - _deltaBetaConstant*puchad);
  if( _relativeIso ) iso /= cand->p4().pt();
  return iso;
}
