/**
  \class    pat::GenMETExtractor GenMETExtractor.h "PhysicsTools/PatAlgos/interface/GenMETExtractor.h"
  \brief    Retrieves the genMET from a pat::MET

   The GenMETExtractor produces the analysis-level pat::MET starting from
   a collection of objects of METType.

  \author   Matthieu Marionneau
  \version  $Id: GenMETExtractor.h,v 1.0 2015/07/22 mmarionn Exp $
*/

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>

namespace pat {

  class GenMETExtractor : public edm::global::EDProducer<> {
  public:
    explicit GenMETExtractor(const edm::ParameterSet &iConfig);
    ~GenMETExtractor() override;

    void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  private:
    edm::EDGetTokenT<std::vector<pat::MET> > metSrcToken_;
  };

}  // namespace pat

using namespace pat;

GenMETExtractor::GenMETExtractor(const edm::ParameterSet &iConfig) {
  edm::InputTag metIT = iConfig.getParameter<edm::InputTag>("metSource");
  metSrcToken_ = consumes<pat::METCollection>(metIT);

  // produces vector of genMet
  produces<std::vector<reco::GenMET> >();
}

GenMETExtractor::~GenMETExtractor() {}

void GenMETExtractor::produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  edm::Handle<std::vector<pat::MET> > src;
  iEvent.getByToken(metSrcToken_, src);
  if (src->empty())
    edm::LogError("GenMETExtractor::produce") << "input genMET collection is empty";

  const reco::GenMET *genMet = src->front().genMET();

  std::vector<reco::GenMET> *genMetCol = new std::vector<reco::GenMET>();
  genMetCol->push_back((*genMet));

  std::unique_ptr<std::vector<reco::GenMET> > genMETs(genMetCol);
  iEvent.put(std::move(genMETs));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenMETExtractor);
