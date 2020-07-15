#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"

class ParameterSetBlobProducer : public edm::global::EDProducer<edm::BeginRunProducer> {
public:
  ParameterSetBlobProducer(edm::ParameterSet const&);

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const final;

  void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const final;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

private:
  edm::EDPutTokenT<std::map<edm::ParameterSetID, edm::ParameterSetBlob>> const token_;
};

ParameterSetBlobProducer::ParameterSetBlobProducer(edm::ParameterSet const&)
    : token_{produces<std::map<edm::ParameterSetID, edm::ParameterSetBlob>, edm::Transition::BeginRun>()} {}

void ParameterSetBlobProducer::produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const {}

void ParameterSetBlobProducer::globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&) const {
  std::map<edm::ParameterSetID, edm::ParameterSetBlob> psetMap;
  edm::pset::Registry::instance()->fillMap(psetMap);

  iRun.emplace(token_, std::move(psetMap));
}

DEFINE_FWK_MODULE(ParameterSetBlobProducer);
