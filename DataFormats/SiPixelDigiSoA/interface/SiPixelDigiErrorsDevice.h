#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class SiPixelDigiErrorsDevice : public PortableDeviceCollection<SiPixelDigiErrorsSoA, TDev> {
public:
  SiPixelDigiErrorsDevice() = default;
  template <typename TQueue>
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, TQueue queue)
      : PortableDeviceCollection<SiPixelDigiErrorsSoA, TDev>(maxFedWords, queue), maxFedWords_(maxFedWords) {}

  // Constructor which specifies the SoA size
  explicit SiPixelDigiErrorsDevice(size_t maxFedWords, TDev const& device)
      : PortableDeviceCollection<SiPixelDigiErrorsSoA, TDev>(maxFedWords, device) {}

  auto& error_data() const { return (*this->view().pixelErrors()); }
  auto maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
