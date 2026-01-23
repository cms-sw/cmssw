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
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include <vector>

class HGCalNeighbourFinder {

public:
  HGCalNeighbourFinder(const HGCalDDDConstants*);
  ~HGCalNeighbourFinder() = default;

  std::vector<unsigned int> nearestNeighboursOfDetId (unsigned int) const;

private:
  const HGCalDDDConstants* hgc_;

  // The method edgeIndexForU:(int)iu andV:(int)iv density:(BOOL)HD
  // should not be a Public method in the CMSSW implemention
  //
  int edgeIndexForU(int iu, int iv, bool HD) const;

  int detIdVec[8];
  
  int iuEdgeLD[45];
  int ivEdgeLD[45];
  int sideLD[45];
  
  int iuEdgeHD[69];
  int ivEdgeHD[69];
  int sideHD[69];

  // ---- not relevant for CMSSW implementation
  int combo[6][6];

};

#endif
