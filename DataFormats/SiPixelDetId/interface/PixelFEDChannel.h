#ifndef SiPixelDetId_PixelFEDChannel_H
#define SiPixelDetId_PixelFEDChannel_H

#include "DataFormats/Common/interface/DetSetVectorNew.h"

struct PixelFEDChannel {
  unsigned int fed, link, roc_first, roc_last;
};

inline bool operator<( const PixelFEDChannel& one, const PixelFEDChannel& other) {
  if (one.fed == other.fed) return one.link<other.link;
  return one.fed < other.fed;
}

typedef edmNew::DetSetVector<PixelFEDChannel> PixelFEDChannelCollection;

#endif
