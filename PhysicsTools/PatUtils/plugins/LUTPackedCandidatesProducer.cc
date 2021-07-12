#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include <vector>

class LUTPackedCandidatesProducer : public edm::global::EDProducer<> {
public:
  explicit LUTPackedCandidatesProducer(const edm::ParameterSet &);
  ~LUTPackedCandidatesProducer() override;

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const edm::EDGetTokenT<pat::PackedCandidateCollection> inputCandidates_;

  const unsigned int covSchema_;
  const unsigned int covVersion_;
  const unsigned int nHits_;
  const unsigned int nPixelHits_;
};

//____________________________________________________________________________||
LUTPackedCandidatesProducer::LUTPackedCandidatesProducer(const edm::ParameterSet &iConfig)
    : inputCandidates_(
          consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCandidates"))),
      covSchema_(iConfig.getParameter<unsigned int>("covSchema")),
      covVersion_(iConfig.getParameter<unsigned int>("covVersion")),
      nHits_(iConfig.getParameter<unsigned int>("nHits")),
      nPixelHits_(iConfig.getParameter<unsigned int>("nPixelHits")) {
  produces<pat::PackedCandidateCollection>("");
}

LUTPackedCandidatesProducer::~LUTPackedCandidatesProducer() {}

void LUTPackedCandidatesProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  edm::Handle<pat::PackedCandidateCollection> packedCandidates;
  iEvent.getByToken(inputCandidates_, packedCandidates);

  auto output = std::make_unique<pat::PackedCandidateCollection>();

  for (unsigned int ic = 0, nc = packedCandidates->size(); ic < nc; ++ic) {
    const pat::PackedCandidate &cand = (*packedCandidates)[ic];
    output->push_back(pat::PackedCandidate(cand));

    if (!output->back().hasTrackDetails())
      output->back().setTrackPropertiesLite(covSchema_, covVersion_, nHits_, nPixelHits_);
  }

  iEvent.put(std::move(output));
};

void LUTPackedCandidatesProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("inputCandidates", edm::InputTag("packedPFCandidates"));
  desc.add<unsigned int>("covSchema", 523);
  desc.add<unsigned int>("covVersion", 1);
  desc.add<unsigned int>("nHits", 8);
  desc.add<unsigned int>("nPixelHits", 3);

  descriptions.addWithDefaultLabel(desc);
}

//____________________________________________________________________________||
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LUTPackedCandidatesProducer);
