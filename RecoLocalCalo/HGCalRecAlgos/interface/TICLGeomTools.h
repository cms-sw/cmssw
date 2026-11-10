#ifndef RecoLocalCalo_HGCalRecAlgos_TICLGeomTools_h
#define RecoLocalCalo_HGCalRecAlgos_TICLGeomTools_h

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomSoA.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticlgeom {

  // The device-side detid constants must match the DetId definitions.
  static_assert(detail::kDetOffset == static_cast<uint32_t>(DetId::kDetOffset));
  static_assert(detail::kHGCalEE == DetId::HGCalEE && detail::kHGCalHSi == DetId::HGCalHSi &&
                detail::kHGCalHSc == DetId::HGCalHSc);
  static_assert(kUnknownSensorGroup == hgcal::UNKNOWN);
  static_assert(detail::kSiCellBits == HGCSiliconDetId::kHGCalWaferUOffset);
  static_assert(detail::kScCellBits == static_cast<uint32_t>(HGCScintillatorDetId::kHGCalRadiusOffset));

  // Replacement for RecHitTools::getSubdetectorGeometry for the few host
  // consumers that need the actual CaloSubdetectorGeometry (cell corners
  // and the like); they consume CaloGeometry themselves and call this.
  inline const CaloSubdetectorGeometry* getSubdetectorGeometry(const CaloGeometry& geom, const DetId& id) {
    const DetId::Detector det = id.det();
    const int subdet = (det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc)
                           ? ForwardSubdetector::ForwardEmpty
                           : id.subdetId();
    return geom.getSubdetectorGeometry(det, subdet);
  }

  // Host-side drop-in replacement for hgcal::RecHitTools, backed by the
  // TICLGeom SoA product. Methods keep the RecHitTools names and
  // semantics. Detector classification and detid unpacking are computed
  // from the detid bits exactly as RecHitTools does, so they work for any
  // detid; geometry-derived quantities are read from the SoA and require
  // the cell to be part of the configured collection (a miss returns the
  // default value of the ticlgeom::Cell getter). Device code uses TICLGeomSoAConstView.
  class Tools {
  public:
    Tools() = default;

    void setGeometry(TICLGeomHost const& geom) {
      geom_ = geom.const_view();
      // The facade reproduces geometry-type-1 (v9+) layer semantics only;
      // the v8 DetId::Forward and HcalEndcap branches of RecHitTools are
      // deliberately not modelled. Fail loudly if fed a v8 geometry.
      assert(geom_.geometryType() == 1 && "TICLGeom facade supports only geometry type 1 (v9+)");
    }

    int32_t denseId(const DetId& id) const { return cell(id).index(); }

    GlobalPoint getPosition(const DetId& id) const {
      const Position p = cell(id).position();
      return GlobalPoint(p.x, p.y, p.z);
    }

    GlobalPoint getPositionLayer(int layer, bool nose = false, bool barrel = false) const {
      if (barrel) {
        const Position p = geom_.barrelLayerPosition(layer);
        return GlobalPoint(p.x, p.y, 0.f);
      }
      return GlobalPoint(0.f, 0.f, nose ? geom_.noseLayerZ(layer) : geom_.layerZ(layer));
    }

    int zside(const DetId& id) const {
      if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
        return HGCSiliconDetId(id).zside();
      } else if (id.det() == DetId::HGCalHSc) {
        return HGCScintillatorDetId(id).zside();
      } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(ForwardSubdetector::HFNose)) {
        return HFNoseDetId(id).zside();
      } else if (id.det() == DetId::Hcal && id.subdetId() == HcalEndcap) {
        return HcalDetId(id).zside();
      }
      return 0;
    }

    float getSiThickness(const DetId& id) const {
      if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
        return HGCSiliconDetId(id).depletion();
      }
      return isSilicon(id) ? cell(id).thickness() : 0.f;
    }

    int getSiThickIndex(const DetId& id) const {
      if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
        return HGCSiliconDetId(id).type();
      } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(ForwardSubdetector::HFNose)) {
        return HFNoseDetId(id).type();
      }
      return -1;
    }

    float getRadiusToSide(const DetId& id) const {
      return isSilicon(id) ? cell(id).radiusToSide() : std::numeric_limits<float>::max();
    }

    std::pair<float, float> getScintDEtaDPhi(const DetId& id) const {
      if (!isScintillator(id)) {
        return {0.f, 0.f};
      }
      const ScintDEtaDPhi d = cell(id).dEtaDPhi();
      return {d.dEta, d.dPhi};
    }

    unsigned int getLayer(const DetId& id) const {
      if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
        return HGCSiliconDetId(id).layer();
      } else if (id.det() == DetId::HGCalHSc) {
        return HGCScintillatorDetId(id).layer();
      } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(ForwardSubdetector::HFNose)) {
        return HFNoseDetId(id).layer();
      } else if (id.det() == DetId::Hcal && id.subdetId() != HcalEmpty) {
        if (id.subdetId() == HcalBarrel)
          return HcalDetId(id).depth();
        else if (id.subdetId() == HcalOuter)
          return HcalDetId(id).depth() + 1;
        return std::numeric_limits<unsigned int>::max();
      } else if (id.det() == DetId::Ecal) {
        return 0;
      }
      return std::numeric_limits<unsigned int>::max();
    }

    // Layer-count overloads, from the layer scalars (geometry type 1
    // semantics; the v8 DetId::Forward geometries are not reproduced)
    unsigned int getLayer(ForwardSubdetector type) const {
      switch (type) {
        case ForwardSubdetector::HFNose:
          return geom_.noseLastLayer();
        case ForwardSubdetector::ForwardEmpty:
          return geom_.numberOfLayers();
        default:
          return 0;
      }
    }

    unsigned int getLayer(DetId::Detector type, bool nose = false) const {
      switch (type) {
        case DetId::HGCalEE:
          return geom_.lastLayerEE();
        case DetId::HGCalHSi:
          return geom_.lastLayerFH() - geom_.lastLayerEE();
        case DetId::HGCalHSc:
          return geom_.lastLayerBH() - geom_.firstLayerBH() + 1;
        case DetId::Forward:
          return nose ? geom_.noseLastLayer() : geom_.lastLayerFH();
        default:
          return 0;
      }
    }

    unsigned int getLayerWithOffset(const DetId& id) const {
      unsigned int layer = getLayer(id);
      if (id.det() == DetId::HGCalHSi || id.det() == DetId::HGCalHSc) {
        layer += lastLayerEE();
      }
      // HFNose needs no offset; the v8 HcalEndcap offset is not supported
      return layer;
    }

    int getCellType(const DetId& id) const { return cell(id).type(); }

    int getSensorGroup(const DetId& id) const { return cell(id).sensorGroup(); }

    std::pair<int, int> getWafer(const DetId& id) const {
      if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
        const HGCSiliconDetId hid(id);
        return {hid.waferU(), hid.waferV()};
      } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(ForwardSubdetector::HFNose)) {
        const HFNoseDetId hid(id);
        return {hid.waferU(), hid.waferV()};
      }
      return {kInvalidCoord, kInvalidCoord};
    }

    std::pair<int, int> getCell(const DetId& id) const {
      if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
        const HGCSiliconDetId hid(id);
        return {hid.cellU(), hid.cellV()};
      } else if (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(ForwardSubdetector::HFNose)) {
        const HFNoseDetId hid(id);
        return {hid.cellU(), hid.cellV()};
      }
      return {kInvalidCoord, kInvalidCoord};
    }

    bool isHalfCell(const DetId& id) const { return isSilicon(id) && cell(id).isHalfCell(); }

    bool isSilicon(const DetId& id) const {
      return (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi ||
              (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(ForwardSubdetector::HFNose)));
    }

    bool isScintillator(const DetId& id) const { return id.det() == DetId::HGCalHSc; }

    bool isScintillatorFine(const DetId& id) const { return isScintillator(id) && cell(id).isFine(); }

    bool isBarrel(const DetId& id) const {
      return (id.det() == DetId::Ecal && id.subdetId() == EcalBarrel) ||
             (id.det() == DetId::Hcal && id.subdetId() == HcalBarrel) ||
             (id.det() == DetId::Hcal && id.subdetId() == HcalOuter);
    }

    float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const {
      GlobalPoint corrected_position = GlobalPoint(position.x(), position.y(), position.z() - vertex_z);
      return corrected_position.eta();
    }

    float getEta(const DetId& id, const float& vertex_z = 0.) const { return getEta(getPosition(id), vertex_z); }

    float getPhi(const GlobalPoint& position) const { return std::atan2(position.y(), position.x()); }

    // atan2 gives the RecHitTools convention: +pi at the boundary where
    // GlobalPoint::phi wraps to -pi
    float getPhi(const DetId& id) const { return cell(id).phi(); }

    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const {
      return hitEnergy / std::cosh(getEta(position, vertex_z));
    }

    float getPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const {
      return getPt(getPosition(id), hitEnergy, vertex_z);
    }

    int getScintMaxIphi(const DetId& id) const { return isScintillator(id) ? cell(id).maxIphi() : 0; }

    unsigned int lastLayerEE(bool nose = false) const {
      return nose ? HFNoseDetId::HFNoseLayerEEmax : geom_.lastLayerEE();
    }
    unsigned int lastLayerFH() const { return geom_.lastLayerFH(); }
    unsigned int firstLayerBH() const { return geom_.firstLayerBH(); }
    unsigned int lastLayerBH() const { return geom_.lastLayerBH(); }
    unsigned int lastLayer(bool nose = false) const { return nose ? geom_.noseLastLayer() : geom_.numberOfLayers(); }
    unsigned int getNumberOfLayers() const { return geom_.numberOfLayers(); }
    unsigned int lastLayerECAL() const { return geom_.lastLayerECAL(); }
    unsigned int lastLayerBarrel() const { return geom_.lastLayerBarrel(); }
    unsigned int maxNumberOfWafersPerLayer(bool nose = false) const {
      return nose ? geom_.noseMaxWafersPerLayer() : geom_.maxWafersPerLayer();
    }
    int getScintMaxIphi() const { return geom_.scintMaxIphi(); }
    int getGeometryType() const { return geom_.geometryType(); }

    std::pair<uint32_t, uint32_t> firstAndLastLayer(DetId::Detector det, int subdet) const {
      if ((det == DetId::HGCalEE) || ((det == DetId::Forward) && (subdet == ForwardSubdetector::HGCEE))) {
        return {1, lastLayerEE()};
      } else if ((det == DetId::HGCalHSi) || ((det == DetId::Forward) && (subdet == ForwardSubdetector::HGCHEF))) {
        return {lastLayerEE() + 1, lastLayerFH()};
      } else if ((det == DetId::Forward) && (subdet == ForwardSubdetector::HFNose)) {
        return {1, geom_.noseLastLayer()};
      } else {
        return {firstLayerBH(), lastLayerBH()};
      }
    }

    // corners must be the RecHitTools default of 3; the column is
    // precomputed at that value
    bool maskCell(const DetId& id, int corners = 3) const { return cell(id).masked(); }

    hgcal::RecHitTools::siliconWaferInfo getWaferInfo(const DetId& id) const {
      const WaferInfo w = cell(id).waferInfo();
      return {w.type, w.partialType, w.orientation, w.placementIndex, w.cassette};
    }

    hgcal::RecHitTools::scintillatorTileInfo getTileInfo(const DetId& id) const {
      const TileInfo t = cell(id).tileInfo();
      return {t.type, t.sipm, t.cassette};
    }

    // The same geometry for device code.
    TICLGeomSoAConstView const& geom() const { return geom_; }

  private:
    Cell cell(const DetId& id) const { return geom_.cell(id.rawId()); }

    TICLGeomSoAConstView geom_;
  };

}  // namespace ticlgeom

#endif  // RecoLocalCalo_HGCalRecAlgos_TICLGeomTools_h
