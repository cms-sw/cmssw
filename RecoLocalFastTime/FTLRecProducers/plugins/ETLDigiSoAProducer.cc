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
#include "DataFormats/FTLDigiSoA/interface/ETLDigiHostCollection.h"

class ETLDigiSoAProducer : public edm::stream::EDProducer<> {
public:
  explicit ETLDigiSoAProducer(edm::ParameterSet const& ps);

  ~ETLDigiSoAProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& event, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<ETLDigiContentCollection> srcToken_;
  const std::string digiCollectionSoA_;
};

ETLDigiSoAProducer::ETLDigiSoAProducer(edm::ParameterSet const& ps)
    : srcToken_(consumes<ETLDigiContentCollection>(ps.getParameter<edm::InputTag>("etlDigiCollection"))),
      digiCollectionSoA_(ps.getParameter<std::string>("digiCollectionSoATag")) {
  produces<etldigi::ETLDigiHostCollection>(digiCollectionSoA_);
}

void ETLDigiSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("etlDigiCollection", edm::InputTag("mix", "MTDEndcap"));
  desc.add<std::string>("digiCollectionSoATag", "MTDEndcapSoA");
  descriptions.add("etlDigiSoAProducer", desc);
}

void ETLDigiSoAProducer::produce(edm::Event& event, edm::EventSetup const&) {
  // get input AoS collection
  auto const& aos = event.get(srcToken_);

  // allocate SoA
  auto soa = std::make_unique<etldigi::ETLDigiHostCollection>(cms::alpakatools::host(), aos.size());
  auto view = soa->view();

  LogTrace("ETLDigiSoAProducer") << "Converting ETLDigiContentCollection of size = " << aos.size();

  // copy fields
  for (size_t i = 0; i < aos.size(); ++i) {
    view.rawId()[i] = aos[i].krawId();
    view.header()[i] = aos[i].kheader();
    view.status()[i] = aos[i].kstatus();
    view.colID()[i] = aos[i].kcolID();
    view.rowID()[i] = aos[i].krowID();
    view.ToAdata()[i] = aos[i].kToAdata();
    view.ToTdata()[i] = aos[i].kToTdata();
    view.CALdata()[i] = aos[i].kCALdata();
  }

#ifdef EDM_ML_DEBUG
  auto const& viewcopy = view;
  if (viewcopy.metadata().size() > 0) {
    LogTrace("ETLDigiSoAProducer") << " ETL Digi SoA collection size = " << viewcopy.metadata().size();

    for (int i = 0; i < viewcopy.metadata().size(); i++) {
      LogTrace("ETLDigiSoAProducer") << "# " << i << " " << viewcopy[i];
    }
  }
#endif

  // put into event
  event.put(std::move(soa), digiCollectionSoA_);
}

// plugin registration
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ETLDigiSoAProducer);
