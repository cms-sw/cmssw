#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClustersHost.h"

// using namespace ::sistrip;
namespace sistrip {
  class SiStripClustersToLegacy : public edm::global::EDProducer<> {
  public:
    SiStripClustersToLegacy(edm::ParameterSet const& iConfig)
        : siStripClustersToken_{consumes(iConfig.getParameter<edm::InputTag>("source"))},
          siStripClustersSetVecPutToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
      auto const& clusters_onHost = iEvent.get(siStripClustersToken_);

      const auto clusterSize = clusters_onHost->clusterSize();
      const auto clusterADCs = clusters_onHost->clusterADCs();
      const auto detIDs = clusters_onHost->clusterDetId();
      const auto stripIDs = clusters_onHost->firstStrip();
      const auto trueCluster = clusters_onHost->trueCluster();

      // Educated guess for the total number of detector IDs,
      // based on Run: 386593 Event: 536278171 with 13883 detectors.
      const unsigned int kInitSeedStripsSize = 15000;
      // The number of clusters from x->nClusters() is an upper limit,
      // the flag trueCluster then mask the real clusters.
      // From Run: 386593 Event: 536278171 there are nClusters=112735 with
      // 99863 real clusters (so 112735-99863 = 12872 clusters are masked out )
      const int nSeedStripsNC = clusters_onHost->nClusters();

      using out_t = edmNew::DetSetVector<SiStripCluster>;
      auto output = std::make_unique<out_t>(edmNew::DetSetVector<SiStripCluster>());
      output->reserve(kInitSeedStripsSize, nSeedStripsNC);

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
      if (edm::isDebugEnabled()) {
        dumpClusters(output.get(), nSeedStripsNC);
      }
      iEvent.put(siStripClustersSetVecPutToken_, std::move(output));
    }

    void dumpClusters(edmNew::DetSetVector<SiStripCluster>* detSetClusters, int clustersPrealloc) const;

  private:
    const edm::EDGetTokenT<SiStripClustersHost> siStripClustersToken_;
    const edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClustersSetVecPutToken_;
  };

  void SiStripClustersToLegacy::dumpClusters(edmNew::DetSetVector<SiStripCluster>* detSetClusters,
                                             int clustersPrealloc) const {
    int clustersAlloc = detSetClusters->dataSize();

    std::ostringstream dumpMsg("[SiStripClustersToLegacy::dumpClusters] Clusters report\n");
    dumpMsg << "Pre-allocated:\t" << clustersPrealloc << "\tProduced:\t" << clustersAlloc << "\n";
    dumpMsg << "   -----  Small cluster dump BEGIN -----   \n";
    dumpMsg << "i\tcIdx\tcSz\tcDetId\tchg\t1st\ttCl\tbary\t - clusterADCs\n";

    int clusterIndex = 0;
    int i = 0;
    for (auto it = detSetClusters->begin(); it != detSetClusters->end(); ++i, it++) {
      if (true || clusterIndex < 50 || clusterIndex > (clustersAlloc - 50) || clusterIndex % 10000 == 0) {
        auto detSet = *it;
        auto cDetId = detSet.detId();
        int clNum = detSet.size();
        for (auto j = detSet.begin(); j != detSet.end(); ++j, ++clusterIndex) {
          auto cluster = *j;
          //
          auto cSz = cluster.size();
          auto chg = cluster.charge();
          auto firstStrip = cluster.firstStrip();
          int trueCluster = 1;
          auto bary = cluster.barycenter();
          dumpMsg << clusterIndex << "\t" << i << "\t" << cSz << "\t" << cDetId << "\t" << chg << "\t" << firstStrip
                  << "\t" << trueCluster << "\t" << bary << "\t - ";
          for (int k = 0; k < (int)cluster.amplitudes().size(); ++k) {
            auto adc = cluster.amplitudes()[k];
            dumpMsg << k << ":" << (int)(adc) << "  ";
          }
          dumpMsg << "\n";
        }
      }
    }
    dumpMsg << "   -----  Small cluster dump END   -----   \n";
    LogDebug("dumpClusters") << dumpMsg.str();
  }

}  // namespace sistrip

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(sistrip::SiStripClustersToLegacy);
