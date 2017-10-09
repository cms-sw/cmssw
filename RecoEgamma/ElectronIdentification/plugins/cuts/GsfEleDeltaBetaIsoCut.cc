#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"


class GsfEleDeltaBetaIsoCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleDeltaBetaIsoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
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
  const reco::GsfElectron::PflowIsolationVariables& pfIso =
    cand->pfIsolationVariables();

  float chad_val   = 0.0;
  float nhad_val   = 0.0;
  float pho_val    = 0.0;
  float puchad_val = 0.0;

  if( _chad_iso.isValid()   && _chad_iso->contains( cand.id() )   &&
      _nhad_iso.isValid()   && _nhad_iso->contains( cand.id() )   && 
      _ph_iso.isValid()     && _ph_iso->contains( cand.id() )     && 
      _PUchad_iso.isValid() && _PUchad_iso->contains( cand.id() )    ) {
    chad_val   = (*_chad_iso)[cand];
    nhad_val   = (*_nhad_iso)[cand];
    pho_val    = (*_ph_iso)[cand];
    puchad_val = (*_PUchad_iso)[cand];
  } else if ( _chad_iso.isValid()   && _chad_iso->idSize()   == 1 &&
              _nhad_iso.isValid()   && _nhad_iso->idSize()   == 1 &&
              _ph_iso.isValid()     && _ph_iso->idSize()     == 1 &&
              _PUchad_iso.isValid() && _PUchad_iso->idSize() == 1 &&
              cand.id() == edm::ProductID() ) {
    // in case we have spoofed a ptr
    //note this must be a 1:1 valuemap (only one product input)
    chad_val   = _chad_iso->begin()[cand.key()];
    nhad_val   = _nhad_iso->begin()[cand.key()];
    pho_val    = _ph_iso->begin()[cand.key()];
    puchad_val = _PUchad_iso->begin()[cand.key()];
  } else if ( _chad_iso.isValid()   && _nhad_iso.isValid()   && 
              _ph_iso.isValid()     && _PUchad_iso.isValid()    ){ // throw an exception
    chad_val   = (*_chad_iso)[cand];
    nhad_val   = (*_nhad_iso)[cand];
    pho_val    = (*_ph_iso)[cand];
    puchad_val = (*_PUchad_iso)[cand];
  }
  
  const float chad   = _chad_iso.isValid()   ? chad_val   : pfIso.sumChargedHadronPt;
  const float nhad   = _nhad_iso.isValid()   ? nhad_val   : pfIso.sumNeutralHadronEt;
  const float pho    = _ph_iso.isValid()     ? pho_val    : pfIso.sumPhotonEt;
  const float puchad = _PUchad_iso.isValid() ? puchad_val : pfIso.sumPUPt;
  float iso = chad + std::max(0.0f, nhad + pho - _deltaBetaConstant*puchad);
  if( _relativeIso ) iso /= cand->p4().pt();
  return iso < isoCut;
}

double GsfEleDeltaBetaIsoCut::value(const reco::CandidatePtr& cand) const {
  edm::Ptr<reco::GsfElectron> ele(cand);
  const reco::GsfElectron::PflowIsolationVariables& pfIso =
    ele->pfIsolationVariables();
  float chad_val   = 0.0;
  float nhad_val   = 0.0;
  float pho_val    = 0.0;
  float puchad_val = 0.0;

  if( _chad_iso.isValid()   && _chad_iso->contains( cand.id() )   &&
      _nhad_iso.isValid()   && _nhad_iso->contains( cand.id() )   &&
      _ph_iso.isValid()     && _ph_iso->contains( cand.id() )     &&
      _PUchad_iso.isValid() && _PUchad_iso->contains( cand.id() )    ) {
    chad_val   = (*_chad_iso)[cand];
    nhad_val   = (*_nhad_iso)[cand];
    pho_val    = (*_ph_iso)[cand];
    puchad_val = (*_PUchad_iso)[cand];
  } else if ( _chad_iso.isValid()   && _chad_iso->idSize()   == 1 &&
              _nhad_iso.isValid()   && _nhad_iso->idSize()   == 1 &&
              _ph_iso.isValid()     && _ph_iso->idSize()     == 1 &&
              _PUchad_iso.isValid() && _PUchad_iso->idSize() == 1 &&
              cand.id() == edm::ProductID() ) {
    // in case we have spoofed a ptr
    //note this must be a 1:1 valuemap (only one product input)
    chad_val   = _chad_iso->begin()[cand.key()];
    nhad_val   = _nhad_iso->begin()[cand.key()];
    pho_val    = _ph_iso->begin()[cand.key()];
    puchad_val = _PUchad_iso->begin()[cand.key()];
  } else if ( _chad_iso.isValid()   && _nhad_iso.isValid()   &&
              _ph_iso.isValid()     && _PUchad_iso.isValid()    ){ // throw an exception
    chad_val   = (*_chad_iso)[cand];
    nhad_val   = (*_nhad_iso)[cand];
    pho_val    = (*_ph_iso)[cand];
    puchad_val = (*_PUchad_iso)[cand];
  }
  
  const float chad   = _chad_iso.isValid()   ? chad_val    : pfIso.sumChargedHadronPt;
  const float nhad   = _nhad_iso.isValid()   ? nhad_val    : pfIso.sumNeutralHadronEt;
  const float pho    = _ph_iso.isValid()     ? pho_val     : pfIso.sumPhotonEt;
  const float puchad = _PUchad_iso.isValid() ? puchad_val : pfIso.sumPUPt;
  float iso = chad + std::max(0.0f, nhad + pho - _deltaBetaConstant*puchad);
  if( _relativeIso ) iso /= cand->p4().pt();
  return iso;
}
