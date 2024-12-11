#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h

#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

class SiPixelDigiErrorsHost : public PortableHostCollection<SiPixelDigiErrorsSoA> {
public:
  SiPixelDigiErrorsHost(edm::Uninitialized) : PortableHostCollection<SiPixelDigiErrorsSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit SiPixelDigiErrorsHost(int maxFedWords, TQueue queue)
      : PortableHostCollection<SiPixelDigiErrorsSoA>(maxFedWords, queue), maxFedWords_(maxFedWords) {}

  int maxFedWords() const { return maxFedWords_; }

  auto& error_data() { return (*view().pixelErrors()); }
  auto const& error_data() const { return (*view().pixelErrors()); }

private:
  int maxFedWords_ = 0;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
