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

  edm::EDGetTokenT<MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi>> srcCSC_;
  edm::EDGetTokenT<L1Phase2MuDTPhContainer> srcDT_;
  edm::EDGetTokenT<L1MuDTChambThContainer> srcDTTheta_;
  edm::EDGetTokenT<RPCDigiCollection> srcRPC_;

  L1TPhase2GMTEndcapStubProcessor* procEndcap_;
  L1TPhase2GMTBarrelStubProcessor* procBarrel_;
  L1TMuon::GeometryTranslator* translator_;
  int verbose_;
};

Phase2L1TGMTStubProducer::Phase2L1TGMTStubProducer(const edm::ParameterSet& iConfig)
    : srcCSC_(
          consumes<MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi>>(iConfig.getParameter<edm::InputTag>("srcCSC"))),
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

  Handle<MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi>> cscDigis;
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
  // gmtStubs
  edm::ParameterSetDescription desc;
  desc.add<int>("verbose", 0);
  desc.add<edm::InputTag>("srcCSC", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("srcDT", edm::InputTag("dtTriggerPhase2PrimitiveDigis"));
  desc.add<edm::InputTag>("srcDTTheta", edm::InputTag("simDtTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("srcRPC", edm::InputTag("simMuonRPCDigis"));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<unsigned int>("verbose", 0);
    psd0.add<int>("minBX", 0);
    psd0.add<int>("maxBX", 0);
    psd0.add<double>("coord1LSB", 0.02453124992);
    psd0.add<double>("eta1LSB", 0.024586688);
    psd0.add<double>("coord2LSB", 0.02453124992);
    psd0.add<double>("eta2LSB", 0.024586688);
    psd0.add<double>("phiMatch", 0.05);
    psd0.add<double>("etaMatch", 0.1);
    desc.add<edm::ParameterSetDescription>("Endcap", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("verbose", 0);
    psd0.add<int>("minPhiQuality", 0);
    psd0.add<int>("minThetaQuality", 0);
    psd0.add<int>("minBX", 0);
    psd0.add<int>("maxBX", 0);
    psd0.add<double>("phiLSB", 0.02453124992);
    psd0.add<int>("phiBDivider", 16);
    psd0.add<double>("etaLSB", 0.024586688);
    psd0.add<std::vector<int>>(
        "eta_1",
        {
            -46, -45, -43, -41, -39, -37, -35, -30, -28, -26, -23, -20, -18, -15, -9, -6, -3, -1,
            1,   3,   6,   9,   15,  18,  20,  23,  26,  28,  30,  35,  37,  39,  41, 43, 45, 1503,
        });
    psd0.add<std::vector<int>>(
        "eta_2",
        {
            -41, -39, -38, -36, -34, -32, -30, -26, -24, -22, -20, -18, -15, -13, -8, -5, -3, -1,
            1,   3,   5,   8,   13,  15,  18,  20,  22,  24,  26,  30,  32,  34,  36, 38, 39, 1334,
        });
    psd0.add<std::vector<int>>(
        "eta_3",
        {
            -35, -34, -32, -31, -29, -27, -26, -22, -20, -19, -17, -15, -13, -11, -6, -4, -2, -1,
            1,   2,   4,   6,   11,  13,  15,  17,  19,  20,  22,  26,  27,  29,  31, 32, 34, 1148,
        });
    psd0.add<std::vector<int>>("coarseEta_1",
                               {
                                   0,
                                   23,
                                   41,
                               });
    psd0.add<std::vector<int>>("coarseEta_2",
                               {
                                   0,
                                   20,
                                   36,
                               });
    psd0.add<std::vector<int>>("coarseEta_3",
                               {
                                   0,
                                   17,
                                   31,
                               });
    psd0.add<std::vector<int>>("coarseEta_4",
                               {
                                   0,
                                   14,
                                   27,
                               });
    psd0.add<std::vector<int>>("phiOffset",
                               {
                                   1,
                                   0,
                                   0,
                                   0,
                               });
    desc.add<edm::ParameterSetDescription>("Barrel", psd0);
  }
  descriptions.add("gmtStubs", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTStubProducer);
