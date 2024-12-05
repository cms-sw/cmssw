#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigisHost_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigisHost_h

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class SiPixelDigisHost : public PortableHostCollection<SiPixelDigisSoA> {
public:
  SiPixelDigisHost(edm::Uninitialized) : PortableHostCollection<SiPixelDigisSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit SiPixelDigisHost(size_t maxFedWords, TQueue queue)
      : PortableHostCollection<SiPixelDigisSoA>(maxFedWords + 1, queue) {}

  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;
};

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigisHost_h
