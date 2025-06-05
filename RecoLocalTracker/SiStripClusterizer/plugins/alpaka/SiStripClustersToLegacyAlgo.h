#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripClustersToLegacyAlgo_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripClustersToLegacyAlgo_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"
#include "DataFormats/SiStripDigiSoA/interface/SiStripDigiHost.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistripConverter {
  GENERATE_SOA_LAYOUT(SiStripClusterrSlimSoALayout,
                      SOA_COLUMN(uint32_t, clusterIndex),
                      SOA_COLUMN(uint16_t, clusterSize),
                      SOA_COLUMN(uint32_t, clusterDetId),
                      SOA_COLUMN(uint16_t, firstStrip),
                      SOA_COLUMN(float, barycenter),
                      SOA_COLUMN(float, charge),
                      //
                      SOA_SCALAR(uint32_t, nClusters),
                      SOA_SCALAR(uint32_t, maxClusterSize))

  using SiStripClustersSlimSoA = SiStripClusterrSlimSoALayout<>;
  using SiStripClustersSlimView = SiStripClustersSlimSoA::View;

  using SiStripClustersSlimDevice = PortableCollection<SiStripClustersSlimSoA>;

  using SiStripClustersSlimHost = PortableHostCollection<SiStripClustersSlimSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistripConverter

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  using namespace sistripConverter;

  class SiStripClustersToLegacyAlgo {
  public:
    SiStripClustersToLegacyAlgo() = default;
    ~SiStripClustersToLegacyAlgo() = default;

    void consumeSoA(Queue& queue, const SiStripClustersDevice& clusters_d, uint32_t goodCandidates);
    std::unique_ptr<edmNew::DetSetVector<SiStripCluster>> convert(Queue& queue, const SiStripDigiHost& amplitudes_h);

  private:
    const SiStripClustersDevice* clusters_d_ = nullptr;
    uint32_t goodCandidates_ = 0;

    std::optional<SiStripClustersSlimHost> clusters_h_;

    void dumpClusters(edmNew::DetSetVector<SiStripCluster>* detSetClusters) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#endif
