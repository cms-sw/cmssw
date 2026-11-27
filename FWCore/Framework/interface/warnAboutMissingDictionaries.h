#ifndef FWCore_Framework_warnAboutMissingDictionaries_h
#define FWCore_Framework_warnAboutMissingDictionaries_h

#include "DataFormats/Provenance/interface/ProductDescriptionFwd.h"

namespace edm {
  // Warns, once per product type, about the classes of a persistent product
  // that have no ROOT data dictionary, for which ROOT falls back to cling or
  // streamer emulation, with a significant performance cost.
  void warnAboutMissingDictionaries(ProductDescription const& prod);
}  // namespace edm

#endif
