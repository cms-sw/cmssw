
#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "SiStripClustersToLegacyAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  using namespace sistripConverter;
  using namespace cms::alpakatools;

  class SiStripConvKer_fillClCollSlim {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_HOST_ACC void operator()(TAcc const& acc,
                                       const uint32_t nbClusterCandidates,
                                       SiStripClusterConstView fullCollection,
                                       SiStripClustersSlimView slimCollection) const {
      //
      // for (uint32_t i=0; i<nbClusterCandidates; ++i) {
      for (auto i : uniform_elements(acc, nbClusterCandidates)) {
        // if (i<100 || i> (nbClusterCandidates - 100)) {
        //   printf(
        //     "#fccs,%i,%i,%i,%i,%i,%i,%f,%f,%i\n",
        //     i,
        //     fullCollection.clusterIndex(i),
        //     fullCollection.clusterSize(i),
        //     fullCollection.clusterDetId(i),
        //     fullCollection.firstStrip(i),
        //     fullCollection.candidateAccepted(i),
        //     fullCollection.barycenter(i),
        //     fullCollection.charge(i),
        //     fullCollection.candidateAcceptedPrefix(i)
        //   );
        // }
        const auto full = fullCollection[i];
        if (full.candidateAccepted()) {
          const uint32_t idx = full.candidateAcceptedPrefix() - 1;
          slimCollection[idx] = {
            full.clusterIndex(),
            full.clusterSize(),
            full.clusterDetId(),
            full.firstStrip(),
            full.barycenter(),
            full.charge()
          };
        }
      }

      if (once_per_grid(acc)) {
        const uint32_t realClusters = fullCollection.candidateAcceptedPrefix(fullCollection.metadata().size() - 1);
        slimCollection.nClusters() = realClusters;
        slimCollection.maxClusterSize() = fullCollection.maxClusterSize();
      }
    }
  };

  void SiStripClustersToLegacyAlgo::consumeSoA(Queue& queue,
                                               const SiStripClusterDevice& clusters_d,
                                               uint32_t nbClusters,
                                               uint32_t nbClusterCandidates) {
    // // Store pointers to the clusters and amplitudes
    // clusters_d_ = &clusters_d;

    // Number of candidates and actual clusters
    goodCandidates_ = nbClusters;
    const uint32_t clusterCandidates = nbClusterCandidates;

    // std::ostringstream msg;
    // const uint32_t clusterCollectionSize = clusters_d.view().metadata().size();
    // msg << "#csSoA," << clusterCollectionSize << ","  << clusterCandidates << "," << goodCandidates_;
    // std::cout << msg.str() << std::endl;

    // Prepare a new cluster collection with the size of goodCandidates
    auto clusters_d_slim = SiStripClustersSlimDevice(goodCandidates_, queue);

    // Fill the collection
    uint32_t divider = 64u;
    uint32_t groups = cms::alpakatools::divide_up_by(clusterCandidates, divider);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, divider);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        SiStripConvKer_fillClCollSlim{},
                        clusterCandidates,
                        clusters_d.const_view(),
                        clusters_d_slim.view());

    // Move the clusters and digi to device
    clusters_h_ = std::make_unique<SiStripClustersSlimHost>(nbClusters, queue);
    alpaka::memcpy(queue, clusters_h_->buffer(), clusters_d_slim.buffer());
  }

  std::unique_ptr<edmNew::DetSetVector<SiStripCluster>> SiStripClustersToLegacyAlgo::convert(
      Queue& queue, const SiStripDigiHost& amplitudes_h) {
    const auto clusterIndexArr = (*clusters_h_)->clusterIndex();
    const auto clusterSize = (*clusters_h_)->clusterSize();
    const auto detIDs = (*clusters_h_)->clusterDetId();
    const auto stripIDs = (*clusters_h_)->firstStrip();
    const auto barycenterArr = (*clusters_h_)->barycenter();
    const auto chargeArr = (*clusters_h_)->charge();

    const uint8_t* amplitudesADC = amplitudes_h->adc();

    // alpaka::wait(queue);
    // std::cout << "#cvrt,1" << std::endl;

    // Educated guess for the total number of detector IDs,
    // based on Run: 386593 Event: 536278171 with 13883 detectors.
    // const unsigned int kInitSeedStripsSize = 15000;
    // The number of clusters from x->nClusters() is an upper limit,
    // the flag candidateAccepted then mask the real clusters.
    // From Run: 386593 Event: 536278171 there are nClusters=112735 with
    // 99863 real clusters (so 112735-99863 = 12872 clusters are masked out )
    const int nSeedStripsNC = (*clusters_h_)->nClusters();

    using out_t = edmNew::DetSetVector<SiStripCluster>;
    auto output = std::make_unique<out_t>();
    // output->reserve(kInitSeedStripsSize, nSeedStripsNC);
    // return output;

    for (int i = 0; i < nSeedStripsNC;) {
      const auto detid = detIDs[i];
      out_t::FastFiller record(*output, detid);

      while (i < nSeedStripsNC && detIDs[i] == detid) {
        const auto size = clusterSize[i];
        const auto firstStrip = stripIDs[i];

        const auto index = clusterIndexArr[i];
        std::vector<uint8_t> adcs(amplitudesADC + index, amplitudesADC + index + size);

        const auto barycenter = barycenterArr[i];
        const auto charge = chargeArr[i];

        record.push_back(SiStripCluster(firstStrip, std::move(adcs), barycenter, charge));
        i++;
      }
    }

    if (edm::isDebugEnabled()) {
      dumpClusters(output.get());
    }

    return output;
  }

  void SiStripClustersToLegacyAlgo::dumpClusters(edmNew::DetSetVector<SiStripCluster>* detSetClusters) const {
    int clustersAlloc = detSetClusters->dataSize();

    std::ostringstream dumpMsg("");
    dumpMsg << "#clDump,Produced:" << clustersAlloc << "\n";
    dumpMsg << "i,cIdx,cSz,cDetId,chg,1st,tCl,bary,|clusterADCs|\n";

    int i = 0;
    int cIdx = 0;
    const int trueCluster = 1;
    for (auto it = detSetClusters->begin(); it != detSetClusters->end(); ++cIdx, it++) {
      if (i < 100 || i > (clustersAlloc - 100)) {
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
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip
