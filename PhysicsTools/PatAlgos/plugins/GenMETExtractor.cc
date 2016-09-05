#include "PhysicsTools/PatAlgos/plugins/GenMETExtractor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

using namespace pat;

GenMETExtractor::GenMETExtractor(const edm::ParameterSet& iConfig) {

  edm::InputTag metIT = iConfig.getParameter<edm::InputTag>("metSource");
  metSrcToken_ = consumes<pat::METCollection>(metIT);

  // produces vector of genMet
  produces<std::vector<reco::GenMET> >();
}


GenMETExtractor::~GenMETExtractor() {

}


void GenMETExtractor::produce(edm::StreamID streamID, edm::Event & iEvent,
			      const edm::EventSetup & iSetup) const {

  edm::Handle<std::vector<pat::MET> >  src;
  iEvent.getByToken(metSrcToken_, src);
  if(src->size()==0) edm::LogError("GenMETExtractor::produce") << "input genMET collection is empty" ;

  const reco::GenMET *genMet =	src->front().genMET();
  
  std::vector<reco::GenMET> *genMetCol = new std::vector<reco::GenMET>();
  genMetCol->push_back( (*genMet) );

  std::unique_ptr<std::vector<reco::GenMET> > genMETs(genMetCol);
  iEvent.put(std::move(genMETs));
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenMETExtractor);
