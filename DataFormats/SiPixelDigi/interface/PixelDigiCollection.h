#ifndef TRACKINGOBJECTS_PIXELDIGICOLLECTION_H
#define TRACKINGOBJECTS_PIXELDIGICOLLECTION_H

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include <vector>
#include <map>
#include <utility>

namespace io_v1 {
  class PixelDigiCollection {
  public:
    typedef std::vector<PixelDigi>::const_iterator ContainerIterator;
    typedef std::pair<ContainerIterator, ContainerIterator> Range;
    typedef std::pair<unsigned int, unsigned int> IndexRange;
    typedef std::map<unsigned int, IndexRange> Registry;
    typedef std::map<unsigned int, IndexRange>::const_iterator RegistryIterator;

    PixelDigiCollection() {}

    void put(Range input, unsigned int detID);
    const Range get(unsigned int detID) const;
    const std::vector<unsigned int> detIDs() const;

  private:
    std::vector<PixelDigi> container_;
    Registry map_;
  };
}  // namespace io_v1
using PixelDigiCollection = io_v1::PixelDigiCollection;

#endif  // TRACKINGOBJECTS_PIXELDIGICOLLECTION_H
