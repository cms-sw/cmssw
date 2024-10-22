#ifndef GeometryHGCalGeometry_HGCalMouseBite_h
#define GeometryHGCalGeometry_HGCalMouseBite_h

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

//#define EDM_ML_DEBUG

class HGCalMouseBite {
public:
  HGCalMouseBite(const HGCalDDDConstants& hgc, bool waferRotate = false);
  template <class T>
  bool exclude(const T& id) {
    int iuv = (100 * id.cellU() + id.cellV());
    bool check = ((id.type() == 0) ? (std::binary_search(rejectFine_.begin(), rejectFine_.end(), iuv))
                                   : (std::binary_search(rejectCoarse_.begin(), rejectCoarse_.end(), iuv)));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalMouseBite:: DetId " << id
                                  << " is checked to be in the list of masked ID's with flag " << check;
#endif
    return check;
  }

private:
  std::vector<int> rejectFine_, rejectCoarse_;
};

#endif  // HGCalMouseBite_h
