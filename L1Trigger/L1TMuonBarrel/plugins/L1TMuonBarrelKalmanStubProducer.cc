#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanStubProcessor.h"

//For masks

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

//
// class declaration
//

class L1TMuonBarrelKalmanStubProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TMuonBarrelKalmanStubProducer(const edm::ParameterSet&);
  ~L1TMuonBarrelKalmanStubProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  const edm::EDGetTokenT<L1MuDTChambPhContainer> srcPhi_;
  const edm::EDGetTokenT<L1MuDTChambThContainer> srcTheta_;
  std::unique_ptr<L1TMuonBarrelKalmanStubProcessor> proc_;
  const int verbose_;
  const edm::ESGetToken<L1TMuonBarrelParams, L1TMuonBarrelParamsRcd> bmtfParamsToken_;
  const edm::EDPutTokenT<L1MuKBMTCombinedStubCollection> putToken_;
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
L1TMuonBarrelKalmanStubProducer::L1TMuonBarrelKalmanStubProducer(const edm::ParameterSet& iConfig)
    : srcPhi_(consumes<L1MuDTChambPhContainer>(iConfig.getParameter<edm::InputTag>("srcPhi"))),
      srcTheta_(consumes<L1MuDTChambThContainer>(iConfig.getParameter<edm::InputTag>("srcTheta"))),
      proc_(std::make_unique<L1TMuonBarrelKalmanStubProcessor>(iConfig)),
      verbose_(iConfig.getParameter<int>("verbose")),
      bmtfParamsToken_(esConsumes()),
      putToken_(produces<L1MuKBMTCombinedStubCollection>()) {}

L1TMuonBarrelKalmanStubProducer::~L1TMuonBarrelKalmanStubProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1TMuonBarrelKalmanStubProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  Handle<L1MuDTChambPhContainer> phiIn;
  iEvent.getByToken(srcPhi_, phiIn);

  Handle<L1MuDTChambThContainer> thetaIn;
  iEvent.getByToken(srcTheta_, thetaIn);

  //Get parameters

  const L1TMuonBarrelParams& bmtfParams = iSetup.getData(bmtfParamsToken_);

  L1MuKBMTCombinedStubCollection stubs = proc_->makeStubs(phiIn.product(), thetaIn.product(), bmtfParams);
  if (verbose_ == 1)
    for (const auto& stub : stubs) {
      printf("Stub: wheel=%d sector=%d station =%d tag=%d eta1=%d qeta1=%d eta2=%d qeta2=%d\n",
             stub.whNum(),
             stub.scNum(),
             stub.stNum(),
             stub.tag(),
             stub.eta1(),
             stub.qeta1(),
             stub.eta2(),
             stub.qeta2());
    }

  if (verbose_ == 2) {
    std::cout << "NEW" << std::endl;
    for (uint sector = 0; sector < 12; ++sector)
      proc_->makeInputPattern(phiIn.product(), thetaIn.product(), sector);
  }

  iEvent.emplace(putToken_, std::move(stubs));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void L1TMuonBarrelKalmanStubProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void L1TMuonBarrelKalmanStubProducer::endStream() {}

void L1TMuonBarrelKalmanStubProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonBarrelKalmanStubProducer);
