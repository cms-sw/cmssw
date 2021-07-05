/**
  \class    pat::RecoMETExtractor RecoMETExtractor.h "PhysicsTools/PatAlgos/interface/RecoMETExtractor.h"
  \brief    Retrieves the recoMET from a pat::MET

   The RecoMETExtractor produces the analysis-level pat::MET starting from
   a collection of objects of METType.

  \author   Matthieu Marionneau
  \version  $Id: RecoMETExtractor.h,v 1.0 2015/07/22 mmarionn Exp $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/METReco/interface/MET.h"

#include <memory>

namespace pat {

  class RecoMETExtractor : public edm::global::EDProducer<> {
  public:
    explicit RecoMETExtractor(const edm::ParameterSet& iConfig);
    ~RecoMETExtractor() override;

    void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  private:
    edm::EDGetTokenT<std::vector<pat::MET> > metSrcToken_;

    pat::MET::METCorrectionLevel corLevel_;
  };

}  // namespace pat

using namespace pat;

RecoMETExtractor::RecoMETExtractor(const edm::ParameterSet& iConfig) {
  edm::InputTag metIT = iConfig.getParameter<edm::InputTag>("metSource");
  metSrcToken_ = consumes<pat::METCollection>(metIT);

  std::string corLevel = iConfig.getParameter<std::string>("correctionLevel");

  //all possible met flavors
  if (corLevel == "raw") {
    corLevel_ = pat::MET::Raw;
  } else if (corLevel == "type1") {
    corLevel_ = pat::MET::Type1;
  } else if (corLevel == "type01") {
    corLevel_ = pat::MET::Type01;
  } else if (corLevel == "typeXY") {
    corLevel_ = pat::MET::TypeXY;
  } else if (corLevel == "type1XY") {
    corLevel_ = pat::MET::Type1XY;
  } else if (corLevel == "type01XY") {
    corLevel_ = pat::MET::Type01XY;
  } else if (corLevel == "type1Smear") {
    corLevel_ = pat::MET::Type1Smear;
  } else if (corLevel == "type01Smear") {
    corLevel_ = pat::MET::Type01Smear;
  } else if (corLevel == "type1SmearXY") {
    corLevel_ = pat::MET::Type1SmearXY;
  } else if (corLevel == "type01SmearXY") {
    corLevel_ = pat::MET::Type01SmearXY;
  } else if (corLevel == "rawCalo") {
    corLevel_ = pat::MET::RawCalo;
  } else if (corLevel == "rawDeepResponseTune") {
    corLevel_ = pat::MET::RawDeepResponseTune;
  } else if (corLevel == "rawDeepResolutionTune") {
    corLevel_ = pat::MET::RawDeepResolutionTune;
  } else {
    //throw exception
  }

  // produces vector of recoMet
  produces<std::vector<reco::MET> >();
}

RecoMETExtractor::~RecoMETExtractor() {}

void RecoMETExtractor::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<std::vector<pat::MET> > src;
  iEvent.getByToken(metSrcToken_, src);
  if (src->empty())
    edm::LogError("RecoMETExtractor::produce") << "input reco MET collection is empty";

  std::vector<reco::MET>* metCol = new std::vector<reco::MET>();

  reco::MET met(src->front().corSumEt(corLevel_), src->front().corP4(corLevel_), src->front().vertex());

  metCol->push_back(met);

  std::unique_ptr<std::vector<reco::MET> > recoMETs(metCol);
  iEvent.put(std::move(recoMETs));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoMETExtractor);
