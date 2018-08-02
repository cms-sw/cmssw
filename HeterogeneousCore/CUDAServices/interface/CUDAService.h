#ifndef HeterogeneousCore_CUDAServices_CUDAService_h
#define HeterogeneousCore_CUDAServices_CUDAService_h

#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/StreamID.h"

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ConfigurationDescriptions;
}

/**
 * TODO:
 * - CUDA stream management?
 *   * Not really needed until we want to pass CUDA stream objects from one module to another
 *   * Which is not really needed until we want to go for "streaming mode"
 *   * Until that framework's inter-module synchronization is safe (but not necessarily optimal)
 * - Management of (preallocated) memory?
 */
class CUDAService {
public:
  CUDAService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);
  ~CUDAService();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  // To be used in global context when an edm::Stream is not available
  bool enabled() const { return enabled_; }
  // To be used in stream context when an edm::Stream is available
  bool enabled(edm::StreamID streamId) const { return enabled(static_cast<unsigned int>(streamId)); }
  bool enabled(unsigned int streamId) const { return enabled_ && (numberOfStreamsTotal_ == 0 || streamId < numberOfStreamsTotal_); } // to make testing easier

  int numberOfDevices() const { return numberOfDevices_; }

  // major, minor
  std::pair<int, int> computeCapability(int device) { return computeCapabilities_.at(device); }

  // Returns the id of device with most free memory. If none is found, returns -1.
  int deviceWithMostFreeMemory() const;

  // Set the current device
  void setCurrentDevice(int device) const;

  // Get the current device
  int getCurrentDevice() const;

private:
  int numberOfDevices_ = 0;
  unsigned int numberOfStreamsTotal_ = 0;
  std::vector<std::pair<int, int>> computeCapabilities_;
  bool enabled_ = false;
};

#endif
