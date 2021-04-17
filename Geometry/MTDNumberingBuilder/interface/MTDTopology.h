#ifndef MTDTOPOLOGY_H
#define MTDTOPOLOGY_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include <vector>
#include <string>

class MTDTopology {
public:
  MTDTopology(const int &topologyMode);

  int getMTDTopologyMode() const { return mtdTopologyMode_; }

private:
  const int mtdTopologyMode_;
};

#endif
