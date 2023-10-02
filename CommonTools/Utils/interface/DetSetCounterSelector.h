#ifndef RecoAlgos_DetSetCounterSelector_h
#define RecoAlgos_DetSetCounterSelector_h
/* \class DetSetCounterSelector
 *
 * \author Marco Musich
 *
 * $Id: DetSetCounterSelector.h,v 1.6 2023/09/22 16:00:00 musich Exp $
 */

#include "FWCore/Utilities/interface/TypeDemangler.h"

struct DetSetCounterSelector {
  DetSetCounterSelector(unsigned int minDetSetCounts, unsigned int maxDetSetCounts)
      : minDetSetCounts_(minDetSetCounts), maxDetSetCounts_(maxDetSetCounts) {}
  template <typename T>
  bool operator()(const T& t) const {
#ifdef EDM_ML_DEBUG
    std::string demangledName(edm::typeDemangle(typeid(T).name()));
    edm::LogVerbatim("DetSetCounterSelector") << "counting counts in: " << demangledName << std::endl;
#endif

    // count the number of objects in the DetSet
    unsigned int totalDetSetCounts = t.size();
    return (totalDetSetCounts >= minDetSetCounts_ && totalDetSetCounts <= maxDetSetCounts_);
  }

private:
  unsigned int minDetSetCounts_;
  unsigned int maxDetSetCounts_;
};

#endif
