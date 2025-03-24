#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;

  class SiStripClustersToLegacy : public stream::SynchronizingEDProducer<> {
  public:
    SiStripClustersToLegacy(edm::ParameterSet const& iConfig)
        : SynchronizingEDProducer<>(iConfig),
          siStripClustersToken_{consumes(iConfig.getParameter<edm::InputTag>("source"))},
          siStripClustersSetVecPutToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const& iEvent, device::EventSetup const&) override {}

    void produce(device::Event& iEvent, device::EventSetup const&) override {
      auto const& clusters_onHost = iEvent.get(siStripClustersToken_);

      const int nSeedStripsNC = clusters_onHost->nClusters();
      const auto clusterSize = clusters_onHost->clusterSize();
      const auto clusterADCs = clusters_onHost->clusterADCs();
      const auto detIDs = clusters_onHost->clusterDetId();
      const auto stripIDs = clusters_onHost->firstStrip();
      const auto trueCluster = clusters_onHost->trueCluster();

      const unsigned int initSeedStripsSize = 15000;

      using out_t = edmNew::DetSetVector<SiStripCluster>;
      auto output{std::make_unique<out_t>(edmNew::DetSetVector<SiStripCluster>())};
      output->reserve(initSeedStripsSize, nSeedStripsNC);

      std::vector<uint8_t> adcs;
      for (int i = 0; i < nSeedStripsNC;) {
        const auto detid = detIDs[i];
        out_t::FastFiller record(*output, detid);

        while (i < nSeedStripsNC && detIDs[i] == detid) {
          if (trueCluster[i]) {
            const auto size = clusterSize[i];
            const auto firstStrip = stripIDs[i];

            adcs.clear();
            adcs.reserve(size);

            for (uint32_t j = 0; j < size; ++j) {
              adcs.push_back(clusterADCs[i][j]);
            }
            record.push_back(SiStripCluster(firstStrip, std::move(adcs)));
          }
          i++;
        }
      }

      output->shrink_to_fit();
      iEvent.put(siStripClustersSetVecPutToken_, std::move(output));
    }

  private:
    const edm::EDGetTokenT<SiStripClustersHost> siStripClustersToken_;
    const edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClustersSetVecPutToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(sistrip::SiStripClustersToLegacy);
