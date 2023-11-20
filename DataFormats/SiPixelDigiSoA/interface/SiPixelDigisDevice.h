#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class SiPixelDigisDevice : public PortableDeviceCollection<SiPixelDigisSoA, TDev> {
public:
  SiPixelDigisDevice() = default;
  template <typename TQueue>
  explicit SiPixelDigisDevice(size_t maxFedWords, TQueue queue)
      : PortableDeviceCollection<SiPixelDigisSoA, TDev>(maxFedWords + 1, queue) {}

  // Constructor which specifies the SoA size
  explicit SiPixelDigisDevice(size_t maxFedWords, TDev const &device)
      : PortableDeviceCollection<SiPixelDigisSoA, TDev>(maxFedWords + 1, device) {}

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    nModules_h = nModules;
    nDigis_h = nDigis;
  }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return nDigis_h; }

private:
  uint32_t nModules_h = 0;
  uint32_t nDigis_h = 0;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h
