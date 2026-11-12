#ifndef RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerLaunchParams_h
#define RecoParticleFlow_PFClusterProducer_plugins_alpaka_PFMultiDepthClusterizerLaunchParams_h

#include <algorithm>
#include <array>

enum class KernelLabels : std::size_t {
  ShowerShape,
  ConstructLinks,
  SBlockPrologue,
  MBlockPrologue,
  ECLCC,
  SBlockEpilogue,
  MBlockEpilogue,
  Count
};

class PFMultiDepthClusterizerLaunchParams {
public:
  using DeviceId = std::int32_t;
  using ThreadsPerBlock = std::uint32_t;

  explicit PFMultiDepthClusterizerLaunchParams(std::size_t nDevices) : maxThreadsPerBlockCache_(nDevices) {}

  void Store(DeviceId dev, KernelLabels label, ThreadsPerBlock maxThreads) { params(dev)[index(label)] = maxThreads; }

  [[nodiscard]] ThreadsPerBlock Load(DeviceId dev, KernelLabels label) const {
    const ThreadsPerBlock t = params(dev)[index(label)];
    if (t == unset) {
      throw std::logic_error("Launch params requested before being stored");
    }
    return t;
  }

  [[nodiscard]] ThreadsPerBlock Load(DeviceId dev, KernelLabels label, ThreadsPerBlock requested) const {
    return std::min(requested, Load(dev, label));
  }

private:
  static constexpr std::size_t numKernels = static_cast<std::size_t>(KernelLabels::Count);
  static constexpr ThreadsPerBlock unset = 0;

  using KernelParams = std::array<ThreadsPerBlock, numKernels>;

  static constexpr std::size_t index(KernelLabels label) noexcept {
    const auto i = static_cast<std::size_t>(label);
    assert(i < numKernels && "KernelLabels::Count is not a valid kernel");
    return i;
  }

  KernelParams& params(DeviceId dev) { return maxThreadsPerBlockCache_.at(static_cast<std::size_t>(dev)); }
  const KernelParams& params(DeviceId dev) const { return maxThreadsPerBlockCache_.at(static_cast<std::size_t>(dev)); }

  std::vector<KernelParams> maxThreadsPerBlockCache_;
};

#endif
