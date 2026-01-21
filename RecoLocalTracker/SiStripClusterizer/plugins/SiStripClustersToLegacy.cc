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
#include "DataFormats/SiStripClusterSoA/interface/SiStripClusterHost.h"
#include "DataFormats/SiStripDigiSoA/interface/SiStripDigiHost.h"

// using namespace ::sistrip;
namespace sistrip {
  class SiStripClustersToLegacy : public edm::global::EDProducer<> {
  public:
    SiStripClustersToLegacy(edm::ParameterSet const& iConfig)
        : siStripClustersToken_(consumes(iConfig.getParameter<edm::InputTag>("source"))),
          siStripDigiToken_(consumes(iConfig.getParameter<edm::InputTag>("source"))),
          siStripClustersSetVecPutToken_(produces()) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
      using out_t = edmNew::DetSetVector<SiStripCluster>;
      auto output = std::make_unique<out_t>();

      // Get clusters and amplitudes
      auto const& clusters_onHost = iEvent.get(siStripClustersToken_);
      auto const& amplitudes_onHost = iEvent.get(siStripDigiToken_);

      const uint32_t nClusterCandidates = clusters_onHost->metadata().size();
      const auto& clusterSizeArr = clusters_onHost->clusterSize();
      const auto& detIdArr = clusters_onHost->clusterDetId();
      const auto& firstStripArr = clusters_onHost->firstStrip();
      const auto& candidateAcceptedArr = clusters_onHost->candidateAccepted();
      const auto& clusterIndexArr = clusters_onHost->clusterIndex();
      const auto& barycenterArr = clusters_onHost->barycenter();
      const auto& chargeArr = clusters_onHost->charge();

      const auto& clusterAmplArr = amplitudes_onHost->adc();

      // Return immediately if there are no clusters
      if (nClusterCandidates == 0) {
        iEvent.put(siStripClustersSetVecPutToken_, std::move(output));
        return;
      }

      // Educated guess for the total number of detector IDs,
      // based on Run: 386593 Event: 536278171 with 13883 detectors.
      // const uint32_t nModulesWithClustersGuess = 15000;
      // The number of clusters from x->nClusterCandidates() is an upper limit,
      // the flag trueCluster then mask the real clusters.
      // From Run: 386593 Event: 536278171 there are nClusterCandidates=112735 with
      // 99863 real clusters (so 112735-99863 = 12872 clusters are masked out )
      const uint32_t clusterCandidatesNb = clusters_onHost->nClusterCandidates();
      const uint32_t goodClustersNb = clusters_onHost->candidateAcceptedPrefix(nClusterCandidates - 1);
      // output->reserve(nModulesWithClustersGuess, goodClustersNb);

      uint32_t clusterN = 0;
      for (uint32_t i = 0; i < clusterCandidatesNb && (clusterN < goodClustersNb);) {
        const auto detid = detIdArr[i];
        out_t::FastFiller record(*output, detid);

        while (i < clusterCandidatesNb && detIdArr[i] == detid) {
          if (candidateAcceptedArr[i]) {
            const auto size = clusterSizeArr[i];
            const auto firstStrip = firstStripArr[i];

            const float barycenter = barycenterArr[i];
            const float charge = chargeArr[i];

            const auto index = clusterIndexArr[i];
            auto clusterAdc = clusterAmplArr.subspan(index, size);
            std::vector<uint8_t> adcs(clusterAdc.begin(), clusterAdc.end());

            record.push_back(SiStripCluster(firstStrip, std::move(adcs), barycenter, charge));
            clusterN++;
          }
          i++;
        }
      }

      if (edm::isDebugEnabled()) {
        dumpClusters(output.get());
      }
      iEvent.put(siStripClustersSetVecPutToken_, std::move(output));
    }

    void dumpClusters(edmNew::DetSetVector<SiStripCluster>* detSetClusters) const;

  private:
    const edm::EDGetTokenT<SiStripClusterHost> siStripClustersToken_;
    const edm::EDGetTokenT<SiStripDigiHost> siStripDigiToken_;
    const edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClustersSetVecPutToken_;
  };

  void SiStripClustersToLegacy::dumpClusters(edmNew::DetSetVector<SiStripCluster>* detSetClusters) const {
    int clustersAlloc = detSetClusters->dataSize();

    std::ostringstream dumpMsg;
    dumpMsg << "#clDump,Produced:" << clustersAlloc << "\n";
    dumpMsg << "i,cIdx,cSz,cDetId,chg,1st,tCl,bary,|clusterADCs|\n";

    int i = 0;
    int cIdx = 0;
    const int trueCluster = 1;
    for (auto it = detSetClusters->begin(); it != detSetClusters->end(); ++cIdx, it++) {
      if (true || i < 100 || i > (clustersAlloc - 100)) {
        auto detSet = *it;
        auto cDetId = detSet.detId();
        // int clNum = detSet.size();
        for (auto j = detSet.begin(); j != detSet.end(); ++j, ++i) {
          auto cluster = *j;
          //
          auto cSz = cluster.size();
          auto chg = cluster.charge();
          auto firstStrip = cluster.firstStrip();

          auto bary = cluster.barycenter();

          dumpMsg << i << "," << cIdx << "," << cSz << "," << cDetId << "," << chg << "," << firstStrip << ","
                  << trueCluster << "," << bary << ",|";
          for (int k = 0; k < (int)cluster.amplitudes().size(); ++k) {
            auto adc = cluster.amplitudes()[k];
            dumpMsg << (int)(adc);
            if (k != ((int)cluster.amplitudes().size() - 1)) {
              dumpMsg << "/";
            }
          }
          dumpMsg << "|\n";
        }
      }
    }
    dumpMsg << "#zClDump\n";
    std::cout << dumpMsg.str();
  }

}  // namespace sistrip

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(sistrip::SiStripClustersToLegacy);
