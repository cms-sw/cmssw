#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"


class GsfEleConversionVetoCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleConversionVetoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:  
  edm::Handle<reco::ConversionCollection> _convs;
  edm::Handle<reco::BeamSpot> _thebs;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleConversionVetoCut,
		  "GsfEleConversionVetoCut");

GsfEleConversionVetoCut::GsfEleConversionVetoCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c) {
  edm::InputTag conversiontag = c.getParameter<edm::InputTag>("conversionSrc");
  contentTags_.emplace("conversions",conversiontag);
  edm::InputTag beamspottag = c.getParameter<edm::InputTag>("beamspotSrc");
  contentTags_.emplace("beamspot",beamspottag);
}

void GsfEleConversionVetoCut::setConsumes(edm::ConsumesCollector& cc) {
  auto convs = 
    cc.consumes<reco::ConversionCollection>(contentTags_["conversions"]);
  auto thebs = cc.consumes<reco::BeamSpot>(contentTags_["beamspot"]);
  contentTokens_.emplace("conversions",convs);
  contentTokens_.emplace("beamspot",thebs);
}

void GsfEleConversionVetoCut::getEventContent(const edm::EventBase& ev) {    
  ev.getByLabel(contentTags_["conversions"],_convs);
  ev.getByLabel(contentTags_["beamspot"],_thebs);  
}

CutApplicatorBase::result_type 
GsfEleConversionVetoCut::
operator()(const reco::GsfElectronRef& cand) const{  
  return !ConversionTools::hasMatchedConversion(*cand,_convs,
						_thebs->position());
}
