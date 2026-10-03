#define EDM_ML_DEBUG

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/FTLDigi/interface/MTDDigiCollections.h"
#include "DataFormats/FTLDigiSoA/interface/BTLDigiHostCollection.h"

class BTLDigiSoAProducer : public edm::stream::EDProducer<> {
public:
  explicit BTLDigiSoAProducer(edm::ParameterSet const& ps);

  ~BTLDigiSoAProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& event, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<BTLDigiContentCollection> srcToken_;
  const std::string digiCollectionSoA_;
};

BTLDigiSoAProducer::BTLDigiSoAProducer(edm::ParameterSet const& ps)
    : srcToken_(consumes<BTLDigiContentCollection>(ps.getParameter<edm::InputTag>("btlDigiCollection"))),
      digiCollectionSoA_(ps.getParameter<std::string>("digiCollectionSoATag")) {
  produces<btldigi::BTLDigiHostCollection>(digiCollectionSoA_);
}

void BTLDigiSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("btlDigiCollection", edm::InputTag("mix", "MTDBarrel"));
  desc.add<std::string>("digiCollectionSoATag", "MTDBarrelSoA");
  descriptions.add("btlDigiSoAProducer", desc);
}

void BTLDigiSoAProducer::produce(edm::Event& event, edm::EventSetup const&) {
  // get input AoS collection
  auto const& aos = event.get(srcToken_);

  // allocate SoA
  auto soa = std::make_unique<btldigi::BTLDigiHostCollection>(cms::alpakatools::host(), aos.size());
  auto view = soa->view();

  LogTrace("BTLDigiSoAProducer") << "Converting BTLDigiContentCollection of size = " << aos.size();

  // copy fields
  for (size_t i = 0; i < aos.size(); ++i) {
    view.rawId()[i] = aos[i].krawId();
    view.BC0count()[i] = aos[i].kBC0count();
    view.status()[i] = aos[i].kstatus();
    view.BCcount()[i] = aos[i].kBCcount();
    view.chIDPlus()[i] = aos[i].kchIDPlus();
    view.T1coarsePlus()[i] = aos[i].kT1coarsePlus();
    view.T2coarsePlus()[i] = aos[i].kT2coarsePlus();
    view.EOIcoarsePlus()[i] = aos[i].kEOIcoarsePlus();
    view.ChargePlus()[i] = aos[i].kChargePlus();
    view.T1finePlus()[i] = aos[i].kT1finePlus();
    view.T2finePlus()[i] = aos[i].kT2finePlus();
    view.IdleTimePlus()[i] = aos[i].kIdleTimePlus();
    view.PrevTrigFPlus()[i] = aos[i].kPrevTrigFPlus();
    view.TACIDPlus()[i] = aos[i].kTACIDPlus();
    view.chIDMinus()[i] = aos[i].kchIDMinus();
    view.T1coarseMinus()[i] = aos[i].kT1coarseMinus();
    view.T2coarseMinus()[i] = aos[i].kT2coarseMinus();
    view.EOIcoarseMinus()[i] = aos[i].kEOIcoarseMinus();
    view.ChargeMinus()[i] = aos[i].kChargeMinus();
    view.T1fineMinus()[i] = aos[i].kT1fineMinus();
    view.T2fineMinus()[i] = aos[i].kT2fineMinus();
    view.IdleTimeMinus()[i] = aos[i].kIdleTimeMinus();
    view.PrevTrigFMinus()[i] = aos[i].kPrevTrigFMinus();
    view.TACIDMinus()[i] = aos[i].kTACIDMinus();
  }

#ifdef EDM_ML_DEBUG
  auto const& viewcopy = view;
  if (viewcopy.metadata().size() > 0) {
    LogTrace("BTLDigiSoAProducer") << " BTL Digi SoA collection size = " << viewcopy.metadata().size();

    for (int i = 0; i < viewcopy.metadata().size(); i++) {
      LogTrace("BTLDigiSoAProducer") << "# " << i << " " << viewcopy[i];
    }
  }
#endif

  // put into event
  event.put(std::move(soa), digiCollectionSoA_);
}

// plugin registration
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BTLDigiSoAProducer);
