#ifndef DataFormats_SiStripCluster_interface_SiStripClustersSOA_h
#define DataFormats_SiStripCluster_interface_SiStripClustersSOA_h

#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"

#include <memory>

namespace detail {
  template <typename T>
  using unique_ptr = typename std::unique_ptr<T>;
}

class SiStripClustersSOA : public SiStripClustersSOABase<detail::unique_ptr> {
public:
  SiStripClustersSOA() = default;
  explicit SiStripClustersSOA(uint32_t maxClusters, uint32_t maxStripsPerCluster);
  ~SiStripClustersSOA() override = default;

  SiStripClustersSOA(const SiStripClustersSOA &) = delete;
  SiStripClustersSOA &operator=(const SiStripClustersSOA &) = delete;
  SiStripClustersSOA(SiStripClustersSOA &&) = default;
  SiStripClustersSOA &operator=(SiStripClustersSOA &&) = default;
};

#endif
