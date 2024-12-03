#ifndef MTDTOPOLOGY_H
#define MTDTOPOLOGY_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <vector>
#include <string>

class MTDTopology {
public:
  // number of logical rods, i.e. rows of sensor modules along eta/z in phi, and of modules per rod
  static constexpr size_t nBTLphi = BTLDetId::HALF_ROD * BTLDetId::kModulesPerTrkV2;
  static constexpr size_t nBTLeta =
      2 * BTLDetId::kRUPerTypeV2 * BTLDetId::kCrystalTypes * BTLDetId::kModulesPerRUV2 / BTLDetId::kModulesPerTrkV2;

  struct ETLfaceLayout {
    uint32_t idDiscSide_;  // disc face identifier
    uint32_t idDetType1_;  // module type id identifier for first row

    std::array<std::vector<int>, 2> start_copy_;  // start copy per row, first of type idDetType1_
    std::array<std::vector<int>, 2> offset_;      // offset per row, first of type idDetType1_
  };

  using ETLValues = std::vector<ETLfaceLayout>;

  MTDTopology(const int& topologyMode, const ETLValues& etl);

  int getMTDTopologyMode() const { return mtdTopologyMode_; }

  // BTL topology navigation is based on a predefined order of dets in MTDGeometry, mapped onto phi/eta grid

  std::pair<uint32_t, uint32_t> btlIndex(const uint32_t detId);
  uint32_t btlidFromIndex(const uint32_t iphi, const uint32_t ieta);

  // BTL topology navigation methods, find index of closest module along eta or phi

  size_t phishiftBTL(const uint32_t detid, const int phiShift);
  size_t etashiftBTL(const uint32_t detid, const int etaShift);

  // ETL topology navigation is based on a predefined order of dets in sector

  static bool orderETLSector(const GeomDet*& gd1, const GeomDet*& gd2);

  // navigation methods in ETL topology, provide the index of the det next to DetId for
  // horizontal and vertical shifts in both directions, assuming the predefined order in a sector

  size_t hshiftETL(const uint32_t detid, const int horizontalShift) const;
  size_t vshiftETL(const uint32_t detid, const int verticalShift, size_t& closest) const;

private:
  const int mtdTopologyMode_;

  const ETLValues etlVals_;

  static constexpr size_t failIndex_ =
      std::numeric_limits<unsigned int>::max();  // return out-of-range value for any failure
};

#endif
