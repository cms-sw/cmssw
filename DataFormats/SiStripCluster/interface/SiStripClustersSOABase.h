#ifndef DataFormats_SiStripCluster_interface_SiStripClustersSOABase_
#define DataFormats_SiStripCluster_interface_SiStripClustersSOABase_

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"

#include <cstdint>
#include <limits>

template <template <typename> class T>
class SiStripClustersSOABase {
public:
  //static constexpr uint32_t kClusterMaxStrips = 16;

  SiStripClustersSOABase() = default;
  //explicit SiStripClustersSOABase(uint32_t maxClusters, uint32_t maxStripsPerCluster);
  virtual ~SiStripClustersSOABase() = default;

  SiStripClustersSOABase(const SiStripClustersSOABase&) = delete;
  SiStripClustersSOABase& operator=(const SiStripClustersSOABase&) = delete;
  SiStripClustersSOABase(SiStripClustersSOABase&&) = default;
  SiStripClustersSOABase& operator=(SiStripClustersSOABase&&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_ = nClusters; }
  uint32_t nClusters() const { return nClusters_; }

  void setMaxClusterSize(uint32_t maxClusterSize) { maxClusterSize_ = maxClusterSize; }
  uint32_t maxClusterSize() const { return maxClusterSize_; }

  const auto& clusterIndex() const { return clusterIndex_; }
  const auto& clusterSize() const { return clusterSize_; }
  const auto& clusterADCs() const { return clusterADCs_; }
  const auto& clusterDetId() const { return clusterDetId_; }
  const auto& firstStrip() const { return firstStrip_; }
  const auto& trueCluster() const { return trueCluster_; }
  const auto& barycenter() const { return barycenter_; }
  const auto& charge() const { return charge_; }

  auto& clusterIndex() { return clusterIndex_; }
  auto& clusterSize() { return clusterSize_; }
  auto& clusterADCs() { return clusterADCs_; }
  auto& clusterDetId() { return clusterDetId_; }
  auto& firstStrip() { return firstStrip_; }
  auto& trueCluster() { return trueCluster_; }
  auto& barycenter() { return barycenter_; }
  auto& charge() { return charge_; }

protected:
  T<uint32_t[]> clusterIndex_;
  T<uint32_t[]> clusterSize_;
  T<uint8_t[]> clusterADCs_;
  T<stripgpu::detId_t[]> clusterDetId_;
  T<stripgpu::stripId_t[]> firstStrip_;
  T<bool[]> trueCluster_;
  T<float[]> barycenter_;
  T<float[]> charge_;
  uint32_t nClusters_;
  uint32_t maxClusterSize_;
};
#endif
