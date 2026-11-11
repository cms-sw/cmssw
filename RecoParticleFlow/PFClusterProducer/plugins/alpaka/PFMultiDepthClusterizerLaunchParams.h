#ifndef RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerLaunchParams_h
#define RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerLaunchParams_h

enum class KernelLabels : std::size_t {
  ShowerShape,
  ConstructLinks,
  SBlockPrologue,
  MBlockPrologue,
  ECLCC,
  SBlockEpilogue,
  MBlockEpilogue
};

class PFMultiDepthClusterizerLaunchParams {
public:
  using DeviceId = std::int32_t;
  using ThreadsPerBlock = std::uint32_t;

  PFMultiDepthClusterizerLaunchParams(const std::uint32_t nDevices) : nDevices(nDevices) {}

  void Store(DeviceId dev, KernelLabels kernelLabel, ThreadsPerBlock max_threads) {
    maxThreadsPerBlockCache[dev][kernelLabel] = max_threads;
  }

  [[nodiscard]]
  ThreadsPerBlock Load(DeviceId dev, KernelLabels kernelLabel) const {
    return maxThreadsPerBlockCache.at(dev).at(kernelLabel);
  }

  [[nodiscard]]
  ThreadsPerBlock Load(DeviceId dev, KernelLabels kernelLabel, ThreadsPerBlock threads) const {
    return std::min(threads, maxThreadsPerBlockCache.at(dev).at(kernelLabel));
  }

private:
  const std::uint32_t nDevices;
  std::map<DeviceId, std::map<KernelLabels, ThreadsPerBlock>> maxThreadsPerBlockCache;
};

#endif
