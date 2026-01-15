#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class SiPixelDigiErrorsDevice : public PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA> {
public:
  SiPixelDigiErrorsDevice(edm::Uninitialized)
      : PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, TQueue queue)
      : PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA>(queue, maxFedWords), maxFedWords_(maxFedWords) {}

  // Constructor which specifies the SoA size
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, TDev const& device)
      : PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA>(device, maxFedWords) {}

  auto maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
