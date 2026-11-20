#include <iostream>

#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

// reading MTD tracking rechits, dereferincing the FTLCluster from the OmniCluster, can be used to check exception raised when reading old timing references (OmniCluster v3 w/ isTiming() == true)
class TestReadMTDTrackingRecHits : public edm::one::EDAnalyzer<> {
public:
  explicit TestReadMTDTrackingRecHits(edm::ParameterSet const& config)
      : src_(config.getParameter<edm::InputTag>("src")), token_(consumes<MTDTrackingDetSetVector>(src_)) {}

private:
  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    auto const& hits = event.get(token_);
    if (hits.dataSize() == 0)
      throw cms::Exception("TestReadMTDTrackingRecHits") << "The tracking-rechit collection is empty.";

    std::cout << "[TestReadMTDTrackingRecHits] Loaded event " << event.id() << std::endl;
    for (auto const& detset : hits) {
      for (auto const& hit : detset) {
        auto const& ref = hit.omniCluster();
        if (!ref.isValid() || !ref.isTiming())
          throw cms::Exception("TestReadMTDTrackingRecHits")
              << "A tracking rechit has an invalid or non-timing cluster reference.";

        std::cout << "[TestReadMTDTrackingRecHits] Timing reference: DetId=" << hit.geographicalId().rawId()
                  << " ProductID=" << ref.id() << " index=" << ref.index() << std::endl;
        std::cout
            << "[TestReadMTDTrackingRecHits] About to dereference the OmniCluster as the configured MTD cluster type"
            << std::endl;
        auto const& cluster = ref.mtdCluster();
        std::cout << "[TestReadMTDTrackingRecHits] Dereference returned." << std::endl;
        std::cout << "[TestReadMTDTrackingRecHits] energy=" << cluster.energy() << " time=" << cluster.time()
                  << " timeError=" << cluster.timeError() << std::endl;
        std::cout << "[TestReadMTDTrackingRecHits] Field access completed." << std::endl;
      }
    }
  }

  edm::InputTag const src_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> const token_;
};

DEFINE_FWK_MODULE(TestReadMTDTrackingRecHits);
