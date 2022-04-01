#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "L1Trigger/Phase2L1GMT/interface/L1TPhase2GMTEndcapStubProcessor.h"
#include "L1Trigger/Phase2L1GMT/interface/L1TPhase2GMTBarrelStubProcessor.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
//
// class declaration
//

class Phase2L1TGMTStubProducer : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTStubProducer(const edm::ParameterSet&);
  ~Phase2L1TGMTStubProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  edm::EDGetTokenT<MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi> > srcCSC_;
  edm::EDGetTokenT<L1Phase2MuDTPhContainer> srcDT_;
  edm::EDGetTokenT<L1MuDTChambThContainer> srcDTTheta_;
  edm::EDGetTokenT<RPCDigiCollection> srcRPC_;

  L1TPhase2GMTEndcapStubProcessor* procEndcap_;
  L1TPhase2GMTBarrelStubProcessor* procBarrel_;
  L1TMuon::GeometryTranslator* translator_;
  int verbose_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Phase2L1TGMTStubProducer::Phase2L1TGMTStubProducer(const edm::ParameterSet& iConfig)
    : srcCSC_(
          consumes<MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi> >(iConfig.getParameter<edm::InputTag>("srcCSC"))),
      srcDT_(consumes<L1Phase2MuDTPhContainer>(iConfig.getParameter<edm::InputTag>("srcDT"))),
      srcDTTheta_(consumes<L1MuDTChambThContainer>(iConfig.getParameter<edm::InputTag>("srcDTTheta"))),
      srcRPC_(consumes<RPCDigiCollection>(iConfig.getParameter<edm::InputTag>("srcRPC"))),
      procEndcap_(new L1TPhase2GMTEndcapStubProcessor(iConfig.getParameter<edm::ParameterSet>("Endcap"))),
      procBarrel_(new L1TPhase2GMTBarrelStubProcessor(iConfig.getParameter<edm::ParameterSet>("Barrel"))),
      verbose_(iConfig.getParameter<int>("verbose")) {
  produces<l1t::MuonStubCollection>();
  edm::ConsumesCollector consumesColl(consumesCollector());
  translator_ = new L1TMuon::GeometryTranslator(consumesColl);
}

Phase2L1TGMTStubProducer::~Phase2L1TGMTStubProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  if (procEndcap_ != nullptr)
    delete procEndcap_;
  if (procBarrel_ != nullptr)
    delete procBarrel_;
  if (translator_ != nullptr)
    delete translator_;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1TGMTStubProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  translator_->checkAndUpdateGeometry(iSetup);

  Handle<MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi> > cscDigis;
  iEvent.getByToken(srcCSC_, cscDigis);

  Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByToken(srcRPC_, rpcDigis);

  Handle<L1Phase2MuDTPhContainer> dtDigis;
  iEvent.getByToken(srcDT_, dtDigis);

  Handle<L1MuDTChambThContainer> dtThetaDigis;
  iEvent.getByToken(srcDTTheta_, dtThetaDigis);

  //Generate a unique stub ID
  l1t::MuonStubCollection stubs;

  uint count0 = 0;
  uint count1 = 0;
  uint count2 = 0;
  uint count3 = 0;
  uint count4 = 0;

  l1t::MuonStubCollection stubsEndcap = procEndcap_->makeStubs(*cscDigis, *rpcDigis, translator_, iSetup);
  for (auto& stub : stubsEndcap) {
    if (stub.tfLayer() == 0) {
      stub.setID(count0);
      count0++;
    } else if (stub.tfLayer() == 1) {
      stub.setID(count1);
      count1++;
    } else if (stub.tfLayer() == 2) {
      stub.setID(count2);
      count2++;
    } else if (stub.tfLayer() == 3) {
      stub.setID(count3);
      count3++;
    } else {
      stub.setID(count4);
      count4++;
    }
    stubs.push_back(stub);
  }
  l1t::MuonStubCollection stubsBarrel = procBarrel_->makeStubs(dtDigis.product(), dtThetaDigis.product());
  for (auto& stub : stubsBarrel) {
    if (stub.tfLayer() == 0) {
      stub.setID(count0);
      count0++;
    } else if (stub.tfLayer() == 1) {
      stub.setID(count1);
      count1++;
    } else if (stub.tfLayer() == 2) {
      stub.setID(count2);
      count2++;
    } else if (stub.tfLayer() == 3) {
      stub.setID(count3);
      count3++;
    } else {
      stub.setID(count4);
      count4++;
    }
    stubs.push_back(stub);
  }

  iEvent.put(std::make_unique<l1t::MuonStubCollection>(stubs));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void Phase2L1TGMTStubProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void Phase2L1TGMTStubProducer::endStream() {}

void Phase2L1TGMTStubProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTStubProducer);
