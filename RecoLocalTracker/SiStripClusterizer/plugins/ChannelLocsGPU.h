#ifndef RecoLocalTracker_SiStripClusterizer_plugins_ChannelLocsGPU_h
#define RecoLocalTracker_SiStripClusterizer_plugins_ChannelLocsGPU_h

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"

class ChannelLocsGPU;

template <template <typename> class T>
class ChannelLocsBase {
public:
  ChannelLocsBase(size_t size) : size_(size) {}
  virtual ~ChannelLocsBase() = default;

  ChannelLocsBase(ChannelLocsBase&& arg)
      : input_(std::move(arg.input_)),
        inoff_(std::move(arg.inoff_)),
        offset_(std::move(arg.offset_)),
        length_(std::move(arg.length_)),
        fedID_(std::move(arg.fedID_)),
        fedCh_(std::move(arg.fedCh_)),
        detID_(std::move(arg.detID_)),
        size_(arg.size_) {}

  void setChannelLoc(uint32_t index,
                     const uint8_t* input,
                     size_t inoff,
                     size_t offset,
                     uint16_t length,
                     stripgpu::fedId_t fedID,
                     stripgpu::fedCh_t fedCh,
                     stripgpu::detId_t detID) {
    input_[index] = input;
    inoff_[index] = inoff;
    offset_[index] = offset;
    length_[index] = length;
    fedID_[index] = fedID;
    fedCh_[index] = fedCh;
    detID_[index] = detID;
  }

  size_t size() const { return size_; }

  const uint8_t* input(uint32_t index) const { return input_[index]; }
  size_t inoff(uint32_t index) const { return inoff_[index]; }
  size_t offset(uint32_t index) const { return offset_[index]; }
  uint16_t length(uint32_t index) const { return length_[index]; }
  stripgpu::fedId_t fedID(uint32_t index) const { return fedID_[index]; }
  stripgpu::fedCh_t fedCh(uint32_t index) const { return fedCh_[index]; }
  stripgpu::detId_t detID(uint32_t index) const { return detID_[index]; }

  const uint8_t* const* input() const { return input_.get(); }
  size_t* inoff() const { return inoff_.get(); }
  size_t* offset() const { return offset_.get(); }
  uint16_t* length() const { return length_.get(); }
  stripgpu::fedId_t* fedID() const { return fedID_.get(); }
  stripgpu::fedCh_t* fedCh() const { return fedCh_.get(); }
  stripgpu::detId_t* detID() const { return detID_.get(); }

protected:
  T<const uint8_t*[]> input_;  // input raw data for channel
  T<size_t[]> inoff_;          // offset in input raw data
  T<size_t[]> offset_;         // global offset in alldata
  T<uint16_t[]> length_;       // length of channel data
  T<stripgpu::fedId_t[]> fedID_;
  T<stripgpu::fedCh_t[]> fedCh_;
  T<stripgpu::detId_t[]> detID_;
  size_t size_ = 0;
};

class ChannelLocs : public ChannelLocsBase<cms::cuda::host::unique_ptr> {
  friend class ChannelLocsGPU;

public:
  ChannelLocs(size_t size, cudaStream_t stream);
  ChannelLocs(ChannelLocs&& arg) : ChannelLocsBase(std::move(arg)) {}

  ChannelLocs(ChannelLocs&) = delete;
  ChannelLocs(const ChannelLocs&) = delete;
  ChannelLocs& operator=(const ChannelLocs&) = delete;
  ChannelLocs& operator=(ChannelLocs&&) = delete;

  ~ChannelLocs() override = default;
};

class ChannelLocsView {
public:
  void fill(const ChannelLocsGPU& c);

  __device__ size_t size() const { return size_; }

  __device__ const uint8_t* input(uint32_t index) const { return input_[index]; }
  __device__ size_t inoff(uint32_t index) const { return inoff_[index]; }
  __device__ size_t offset(uint32_t index) const { return offset_[index]; }
  __device__ uint16_t length(uint32_t index) const { return length_[index]; }
  __device__ stripgpu::fedId_t fedID(uint32_t index) const { return fedID_[index]; }
  __device__ stripgpu::fedCh_t fedCh(uint32_t index) const { return fedCh_[index]; }
  __device__ stripgpu::detId_t detID(uint32_t index) const { return detID_[index]; }

private:
  const uint8_t* const* input_;  // input raw data for channel
  size_t* inoff_;                // offset in input raw data
  size_t* offset_;               // global offset in alldata
  uint16_t* length_;             // length of channel data
  stripgpu::fedId_t* fedID_;
  stripgpu::fedCh_t* fedCh_;
  stripgpu::detId_t* detID_;
  size_t size_;
};

class ChannelLocsGPU : public ChannelLocsBase<cms::cuda::device::unique_ptr> {
public:
  //using Base = ChannelLocsBase<cms::cuda::device::unique_ptr>;
  ChannelLocsGPU(size_t size, cudaStream_t stream);
  ChannelLocsGPU(ChannelLocsGPU&& arg)
      : ChannelLocsBase(std::move(arg)), channelLocsViewGPU_(std::move(arg.channelLocsViewGPU_)) {}

  ChannelLocsGPU(ChannelLocsGPU&) = delete;
  ChannelLocsGPU(const ChannelLocsGPU&) = delete;
  ChannelLocsGPU& operator=(const ChannelLocsGPU&) = delete;
  ChannelLocsGPU& operator=(ChannelLocsGPU&&) = delete;

  ~ChannelLocsGPU() override = default;

  void setVals(const ChannelLocs* c, cms::cuda::host::unique_ptr<const uint8_t*[]> inputGPU, cudaStream_t stream);
  const ChannelLocsView* channelLocsView() const { return channelLocsViewGPU_.get(); }

private:
  cms::cuda::device::unique_ptr<ChannelLocsView> channelLocsViewGPU_;
};

#endif
