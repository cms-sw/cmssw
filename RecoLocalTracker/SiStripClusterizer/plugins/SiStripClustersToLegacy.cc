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
      auto const& clusters_onHost = iEvent.get(siStripClustersToken_);
      auto const& amplitudes_onHost = iEvent.get(siStripDigiToken_);

      const uint32_t clustersArrSize = clusters_onHost->metadata().size();
      const auto clusterSizeArr = clusters_onHost->clusterSize();
      const auto detIdArr = clusters_onHost->clusterDetId();
      const auto firstStripArr = clusters_onHost->firstStrip();
      const auto candidateAcceptedArr = clusters_onHost->candidateAccepted();
      const uint32_t* clusterIndexArr = clusters_onHost->clusterIndex();
      const float* barycenterArr = clusters_onHost->barycenter();
      const float* chargeArr = clusters_onHost->charge();

      const auto clusterAmplArr = amplitudes_onHost->adc();

      // Educated guess for the total number of detector IDs,
      // based on Run: 386593 Event: 536278171 with 13883 detectors.
      // const uint32_t nModulesWithClustersGuess = 15000;
      // The number of clusters from x->nClusterCandidates() is an upper limit,
      // the flag trueCluster then mask the real clusters.
      // From Run: 386593 Event: 536278171 there are nClusterCandidates=112735 with
      // 99863 real clusters (so 112735-99863 = 12872 clusters are masked out )
      const uint32_t clusterCandidatesNb = clusters_onHost->nClusterCandidates();
      const uint32_t goodClustersNb = clusters_onHost->candidateAcceptedPrefix(clustersArrSize - 1);

      using out_t = edmNew::DetSetVector<SiStripCluster>;
      auto output = std::make_unique<out_t>();
      // output->reserve(nModulesWithClustersGuess, goodClustersNb);

      // Debugging
      // output->reserve(nModulesWithClustersGuess, 10);
      // iEvent.put(siStripClustersSetVecPutToken_, std::move(output));
      // return ;

      // std::vector<uint8_t> adcs;
      uint32_t clusterN = 0;
      for (uint32_t i = 0; i < clusterCandidatesNb && (clusterN < goodClustersNb);) {
        const auto detid = detIdArr[i];
        // std::cout << "#fledm," << i << "," << detid << std::endl;
        out_t::FastFiller record(*output, detid);

        while (clusterN < goodClustersNb && detIdArr[i] == detid) {
          if (candidateAcceptedArr[i]) {
            const auto size = clusterSizeArr[i];
            const auto firstStrip = firstStripArr[i];

            const float barycenter = barycenterArr[i];
            const float charge = chargeArr[i];

            // std::vector<uint8_t> adcs;
            // adcs.reserve(size);

            // for (uint32_t j = 0; j < size; ++j) {
            //   const auto index = clusterIndexArr[i] + j;
            //   const auto adc = clusterAmplArr[index];
            //   adcs.push_back(adc);
            // }

            const auto index = clusterIndexArr[i];
            std::vector<uint8_t> adcs(clusterAmplArr + index, clusterAmplArr + index + size);

            // SiStripCluster(uint16_t firstStrip, std::vector<uint8_t>&& data, float barycenter, float charge)
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

    std::ostringstream dumpMsg("");
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
