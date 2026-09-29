#ifndef HGCalCommonData_HGCalNeighbourFinder_h
#define HGCalCommonData_HGCalNeighbourFinder_h
//
//  HGCalNeighbourFinder.h
//
//  Created by Chris Seez on 25/10/2025.
//  Copyright © 2025 seez. All rights reserved.
//

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include <vector>

class HGCalNeighbourFinder {
public:
  HGCalNeighbourFinder(const HGCalGeometry*);
  ~HGCalNeighbourFinder() = default;

  std::vector<uint32_t> nearestNeighboursOfDetId(uint32_t) const;

private:
  const HGCalGeometry* geom_;
  const HGCalDDDConstants& hgc_;

  // The method edgeIndexForU:(int)iu andV:(int)iv density:(BOOL)HD
  // should not be a Public method in the CMSSW implemention
  //
  int edgeIndexForU(int iu, int iv, bool HD) const;

  uint32_t detIdVec[8];

  uint32_t iuEdgeLD[45];
  uint32_t ivEdgeLD[45];
  uint32_t sideLD[45];

  uint32_t iuEdgeHD[69];
  uint32_t ivEdgeHD[69];
  uint32_t sideHD[69];

  // ---- not relevant for CMSSW implementation
  uint32_t combo[6][6];
};

#endif
