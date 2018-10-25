//STL includes
#include <memory>

//framework includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//other includes
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HFNoseRawToDigiFake : public edm::global::EDProducer<> {
public:
  explicit HFNoseRawToDigiFake(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  edm::EDGetTokenT<HGCalDigiCollection> tok_hfn_;
};

HFNoseRawToDigiFake::HFNoseRawToDigiFake(const edm::ParameterSet& iConfig) :
  tok_hfn_(consumes<HGCalDigiCollection>(iConfig.getParameter<edm::InputTag>("hfnoseDigis")))
{
  produces<HGCalDigiCollection>("HFNose");
}

void HFNoseRawToDigiFake::produce(edm::StreamID, edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) const {
  edm::Handle<HGCalDigiCollection> hfn;
  iEvent.getByToken(tok_hfn_,hfn);

  auto out_hfn = std::make_unique<HGCalDigiCollection>();
  if (hfn.isValid()) {
    out_hfn = std::make_unique<HGCalDigiCollection>(*(hfn.product()));
  }
  iEvent.put(std::move(out_hfn),"HFNose");
}

void HFNoseRawToDigiFake::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hfnoseDigis",edm::InputTag("simHFNoseUnsuppressedDigis:HFNose"));
  descriptions.add("HFNoseRawToDigiFake",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFNoseRawToDigiFake);
