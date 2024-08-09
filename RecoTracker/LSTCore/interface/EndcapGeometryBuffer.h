#ifndef RecoTracker_LSTCore_interface_EndcapGeometryBuffers_h
#define RecoTracker_LSTCore_interface_EndcapGeometryBuffers_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "RecoTracker/LSTCore/interface/Constants.h"

namespace lst {

  struct EndcapGeometryDev {
    const unsigned int* geoMapDetId;
    const float* geoMapPhi;

    template <typename TBuff>
    void setData(TBuff const& buf) {
      geoMapDetId = alpaka::getPtrNative(buf.geoMapDetId_buf);
      geoMapPhi = alpaka::getPtrNative(buf.geoMapPhi_buf);
    }
  };

  template <typename TDev>
  struct EndcapGeometryBuffer {
    Buf<TDev, unsigned int> geoMapDetId_buf;
    Buf<TDev, float> geoMapPhi_buf;
    EndcapGeometryDev data_;

    EndcapGeometryBuffer(TDev const& dev, unsigned int nEndCapMap)
        : geoMapDetId_buf(allocBufWrapper<unsigned int>(dev, nEndCapMap)),
          geoMapPhi_buf(allocBufWrapper<float>(dev, nEndCapMap)) {
      data_.setData(*this);
    }

    template <typename TQueue, typename TDevSrc>
    inline void copyFromSrc(TQueue queue, EndcapGeometryBuffer<TDevSrc> const& src) {
      alpaka::memcpy(queue, geoMapDetId_buf, src.geoMapDetId_buf);
      alpaka::memcpy(queue, geoMapPhi_buf, src.geoMapPhi_buf);
    }

    template <typename TQueue, typename TDevSrc>
    EndcapGeometryBuffer(TQueue queue, EndcapGeometryBuffer<TDevSrc> const& src, unsigned int nEndCapMap)
        : EndcapGeometryBuffer(alpaka::getDev(queue), nEndCapMap) {
      copyFromSrc(queue, src);
    }

    inline EndcapGeometryDev const* data() const { return &data_; }
  };

}  // namespace lst

#endif
