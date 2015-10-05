#include "PhysicsTools/PatAlgos/plugins/RecoMETExtractor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

using namespace pat;

RecoMETExtractor::RecoMETExtractor(const edm::ParameterSet& iConfig) {

  edm::InputTag metIT = iConfig.getParameter<edm::InputTag>("metSource");
  metSrcToken_ = consumes<pat::METCollection>(metIT);
  
  std::string corLevel = iConfig.getParameter<std::string>("correctionLevel");

  //all possible met flavors
  if(corLevel=="raw")           { corLevel_=pat::MET::Raw;}
  else if(corLevel=="type1")         { corLevel_=pat::MET::Type1;}
  else if(corLevel=="type01")        { corLevel_=pat::MET::Type01;}
  else if(corLevel=="typeXY")        { corLevel_=pat::MET::TypeXY;}
  else if(corLevel=="type1XY")       { corLevel_=pat::MET::Type1XY;}
  else if(corLevel=="type01XY")      { corLevel_=pat::MET::Type01XY;}
  else if(corLevel=="type1Smear")    { corLevel_=pat::MET::Type1Smear;}
  else if(corLevel=="type01Smear")   { corLevel_=pat::MET::Type01Smear;}
  else if(corLevel=="type1SmearXY")  { corLevel_=pat::MET::Type1SmearXY;}
  else if(corLevel=="type01SmearXY") { corLevel_=pat::MET::Type01SmearXY;}
  else if(corLevel=="rawCalo")       { corLevel_=pat::MET::RawCalo;}
  else {
    //throw exception

  }

  // produces vector of recoMet
  produces<std::vector<reco::MET> >();
}


RecoMETExtractor::~RecoMETExtractor() {

}


void RecoMETExtractor::produce(edm::StreamID streamID, edm::Event & iEvent,
			       const edm::EventSetup & iSetup) const {

  edm::Handle<std::vector<pat::MET> >  src;
  iEvent.getByToken(metSrcToken_, src);
  if(src->size()==0) edm::LogError("RecoMETExtractor::produce") << "input reco MET collection is empty" ;

  std::vector<reco::MET> *metCol = new std::vector<reco::MET>();
  
  reco::MET met(src->front().corP4(corLevel_), src->front().vertex() );
  metCol->push_back( met );
  
  std::auto_ptr<std::vector<reco::MET> > recoMETs(metCol);
  iEvent.put(recoMETs);
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(RecoMETExtractor);
