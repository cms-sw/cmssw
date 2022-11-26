#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SiPixelDigisSoALayout,
                    SOA_COLUMN(int32_t, clus),
                    SOA_COLUMN(uint32_t, pdigi),
                    SOA_COLUMN(uint32_t, rawIdArr),
                    SOA_COLUMN(uint16_t, adc),
                    SOA_COLUMN(uint16_t, xx),
                    SOA_COLUMN(uint16_t, yy),
                    SOA_COLUMN(uint16_t, moduleId))

using SiPixelDigisCUDASOA = SiPixelDigisSoALayout<>;
using SiPixelDigisCUDASOAView = SiPixelDigisCUDASOA::View;
using SiPixelDigisCUDASOAConstView = SiPixelDigisCUDASOA::ConstView;

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class SiPixelDigisCUDA : public cms::cuda::PortableDeviceCollection<SiPixelDigisSoALayout<>> {
public:
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
      : PortableDeviceCollection<SiPixelDigisSoALayout<>>(maxFedWords + 1, stream) {}
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

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

#endif  // CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
