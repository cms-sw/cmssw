#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersHost.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiStripClusters_retriever {
  public:
    SiStripClusters_retriever(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC);
    void makeAsync(device::Event const& iEvent, device::EventSetup const& iSetup);
    sistrip::SiStripClustersHost moveFrom() { return std::move(hostProduct_.value()); }

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    const device::EDGetToken<sistrip::SiStripClustersDevice> getToken_;
    // hold the output product between acquire() and produce()
    std::optional<sistrip::SiStripClustersHost> hostProduct_;
  };

  SiStripClusters_retriever::SiStripClusters_retriever(edm::ParameterSet const& iConfig, edm::ConsumesCollector iC)
      : getToken_(iC.consumes(iConfig.getParameter<edm::InputTag>("source"))) {}

  void SiStripClusters_retriever::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    iDesc.add<edm::InputTag>("source");
  }

  void SiStripClusters_retriever::makeAsync(device::Event const& iEvent, device::EventSetup const& iSetup) {
    sistrip::SiStripClustersDevice const& deviceProduct = iEvent.get(getToken_);
    hostProduct_ = sistrip::SiStripClustersHost{deviceProduct->metadata().size(), iEvent.queue()};
    alpaka::memcpy(iEvent.queue(), hostProduct_->buffer(), deviceProduct.const_buffer());
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiStripClustersToLegacy : public stream::SynchronizingEDProducer<> {
  public:
    SiStripClustersToLegacy(edm::ParameterSet const& iConfig)
        : SynchronizingEDProducer<>(iConfig),
          siStripClustersSetVecPutToken_{produces()},
          helper_{iConfig, consumesCollector()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      SiStripClusters_retriever::fillPSetDescription(desc);
      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      helper_.makeAsync(iEvent, iSetup);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto clusters_onHost = helper_.moveFrom();

#ifdef EDM_ML_DEBUG
      auto clView = clusters_onHost.view();
      LogDebug("SiStripToLegacy") << "Event " << iEvent.id() << " nClusters " << clView.nClusters()
                                  << " maxClustersize " << clView.maxClusterSize() << "\ni\tclIdx\tclSz\tdetID\tchg\n";
      for (unsigned int i = 0; i < clView.nClusters(); ++i) {
        if (i % 100 == 0) {
          LogDebug("SiStripToLegacy") << i << "\t" << clView.clusterIndex(i) << "\t" << clView.clusterSize(i) << "\t"
                                      << clView.clusterDetId(i) << "\t" << clView.charge(i) << "\n";
        }
      }
#endif

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
    const edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClustersSetVecPutToken_;

    SiStripClusters_retriever helper_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiStripClustersToLegacy);
