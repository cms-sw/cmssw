#ifndef RecoHGCal_TICL_Utils_h
#define RecoHGCal_TICL_Utils_h

#include <array>
#include <memory>

#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {
  namespace utils {
    // Build HGCal front disk geometry for track propagation
    // Returns array of [negative-z disk, positive-z disk]
    inline std::array<std::unique_ptr<GeomDet>, 2> buildHGCalFirstDisks(const HGCalDDDConstants& hgcons,
                                                                        const CaloGeometry& geom) {
      hgcal::RecHitTools rhtools;
      rhtools.setGeometry(geom);
      float zVal = hgcons.waferZ(1, true);
      std::pair<float, float> rMinMax = hgcons.rangeR(zVal, true);

      std::array<std::unique_ptr<GeomDet>, 2> firstDisk;
      for (int iSide = 0; iSide < 2; ++iSide) {
        float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
        firstDisk[iSide] = std::make_unique<GeomDet>(
            Disk::build(Disk::PositionType(0, 0, zSide),
                        Disk::RotationType(),
                        SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
                .get());
      }
      return firstDisk;
    }

    inline std::array<std::unique_ptr<GeomDet>, 2> buildHGCalFirstDisks(const HGCalDDDConstants& hgcons) {
      float zVal = hgcons.waferZ(1, true);
      std::pair<float, float> rMinMax = hgcons.rangeR(zVal, true);

      std::array<std::unique_ptr<GeomDet>, 2> firstDisk;
      for (int iSide = 0; iSide < 2; ++iSide) {
        float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
        firstDisk[iSide] = std::make_unique<GeomDet>(
            Disk::build(Disk::PositionType(0, 0, zSide),
                        Disk::RotationType(),
                        SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
                .get());
      }
      return firstDisk;
    }

    inline std::array<std::unique_ptr<GeomDet>, 2> buildHGCalInterfaceDisks(const HGCalDDDConstants& hgcons,
                                                                            const hgcal::RecHitTools& rhtools) {
      float zVal_interface = rhtools.getPositionLayer(rhtools.lastLayerEE()).z();
      std::pair<float, float> rMinMax_interface = hgcons.rangeR(zVal_interface, true);

      std::array<std::unique_ptr<GeomDet>, 2> interfaceDisk;
      for (int iSide = 0; iSide < 2; ++iSide) {
        float zSide = (iSide == 0) ? (-1. * zVal_interface) : zVal_interface;
        interfaceDisk[iSide] = std::make_unique<GeomDet>(
            Disk::build(Disk::PositionType(0, 0, zSide),
                        Disk::RotationType(),
                        SimpleDiskBounds(rMinMax_interface.first, rMinMax_interface.second, zSide - 0.5, zSide + 0.5))
                .get());
      }
      return interfaceDisk;
    }
  }  // namespace utils
}  // namespace ticl

#endif  // RecoHGCal_TICL_Utils_h
