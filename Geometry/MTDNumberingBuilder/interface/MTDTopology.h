#ifndef MTDTOPOLOGY_H
#define MTDTOPOLOGY_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include <vector>
#include <string>

class MTDTopology {
public:
  struct ETLfaceLayout {
    int idDiscSide_;
    int idDetType1_;

    std::vector<int> start_copy_1_;
    std::vector<int> start_copy_2_;
    std::vector<int> offset_1_;
    std::vector<int> offset_2_;
  };

  using ETLValues = std::vector<ETLfaceLayout>;

  MTDTopology(const int& topologyMode, const ETLValues& etl);

  int getMTDTopologyMode() const { return mtdTopologyMode_; }

private:
  const int mtdTopologyMode_;

  const ETLValues etlVals_;
};

#endif
