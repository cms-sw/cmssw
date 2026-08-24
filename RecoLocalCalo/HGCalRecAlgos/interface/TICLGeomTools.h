#ifndef RecoLocalCalo_HGCalRecAlgos_TICLGeomTools_h
#define RecoLocalCalo_HGCalRecAlgos_TICLGeomTools_h

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticlgeom {

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
  // TICLGeom SoA products. Methods keep the RecHitTools names and
  // semantics. Detector classification and detid unpacking are computed
  // from the detid bits exactly as RecHitTools does, so they work for any
  // detid; geometry-derived quantities are read from the SoA and require
  // the cell to be part of the configured collection (a miss returns the
  // sentinel of the corresponding column). Device code reads the same
  // columns directly.
  class Tools {
  public:
    Tools() = default;

    void setGeometry(TICLGeomHost const& geom, TICLGeomLookupHost const& lookup, TICLGeomLayersHost const& layers) {
      auto const& view = geom.const_view();
      common_ = view.common();
      silicon_ = view.silicon();
      scint_ = view.scint();
      lookup_ = lookup.const_view();
      layers_ = layers.const_view();
      // The facade reproduces geometry-type-1 (v9+) layer semantics only;
      // the v8 DetId::Forward and HcalEndcap branches of RecHitTools are
      // deliberately not modelled. Fail loudly if fed a v8 geometry.
      assert(common_.geometryType() == 1 && "TICLGeom facade supports only geometry type 1 (v9+)");
    }

    int32_t denseId(const DetId& id) const { return denseIdOf(lookup_, common_, id.rawId()); }

    // block-local indices; the cells are ordered [barrel|silicon|scint]
    int32_t siliconLocal(int32_t i) const { return i - common_.nBarrel(); }
    int32_t scintLocal(int32_t i) const { return i - common_.nBarrel() - common_.nSilicon(); }

    GlobalPoint getPosition(const DetId& id) const {
      const int32_t i = denseId(id);
      if (i < 0) {
        return GlobalPoint();
      }
      return GlobalPoint(common_[i].x(), common_[i].y(), common_[i].z());
    }

    GlobalPoint getPositionLayer(int layer, bool nose = false, bool barrel = false) const {
      const int32_t lay = std::abs(layer);
      if (barrel) {
        if (lay < layers_.metadata().size()) {
          return GlobalPoint(layers_[lay].barrelX(), layers_[lay].barrelY(), 0.f);
        }
        return GlobalPoint();
      }
      if (lay >= layers_.metadata().size()) {
        return GlobalPoint();
      }
      const float z = nose ? layers_[lay].noseZ() : layers_[lay].z();
      return GlobalPoint(0.f, 0.f, (layer > 0) ? z : -z);
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
      if (!isSilicon(id)) {
        return 0.f;
      }
      const int32_t i = denseId(id);
      return (i >= 0) ? silicon_[siliconLocal(i)].siThickness() : 0.f;
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
      if (!isSilicon(id)) {
        return std::numeric_limits<float>::max();
      }
      const int32_t i = denseId(id);
      return (i >= 0) ? silicon_[siliconLocal(i)].radiusToSide() : std::numeric_limits<float>::max();
    }

    std::pair<float, float> getScintDEtaDPhi(const DetId& id) const {
      if (!isScintillator(id)) {
        return {0.f, 0.f};
      }
      const int32_t i = denseId(id);
      const int32_t j = (i >= 0) ? scintLocal(i) : -1;
      return (j >= 0) ? std::pair<float, float>{scint_[j].scintDEta(), scint_[j].scintDPhi()}
                      : std::pair<float, float>{0.f, 0.f};
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
        case ForwardSubdetector::HGCEE:
          return common_.lastLayerEE();
        case ForwardSubdetector::HGCHEF:
          return common_.lastLayerFH() - common_.lastLayerEE();
        case ForwardSubdetector::HGCHEB:
          return common_.lastLayerBH() - common_.firstLayerBH() + 1;
        case ForwardSubdetector::HFNose:
          return common_.noseLastLayer();
        case ForwardSubdetector::ForwardEmpty:
          return common_.numberOfLayers();
        default:
          return 0;
      }
    }

    unsigned int getLayer(DetId::Detector type, bool nose = false) const {
      switch (type) {
        case DetId::HGCalEE:
          return common_.lastLayerEE();
        case DetId::HGCalHSi:
          return common_.lastLayerFH() - common_.lastLayerEE();
        case DetId::HGCalHSc:
          return common_.lastLayerBH() - common_.firstLayerBH() + 1;
        case DetId::Forward:
          return nose ? common_.noseLastLayer() : common_.lastLayerFH();
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

    int getCellType(const DetId& id) const {
      const int32_t i = denseId(id);
      return (i >= 0) ? common_[i].cellType() : -1;
    }

    int getSensorGroup(const DetId& id) const {
      const int32_t i = denseId(id);
      return (i >= 0) ? common_[i].sensorGroup() : static_cast<int>(hgcal::UNKNOWN);
    }

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

    bool isHalfCell(const DetId& id) const {
      if (!isSilicon(id)) {
        return false;
      }
      const int32_t i = denseId(id);
      return (i >= 0) ? silicon_[siliconLocal(i)].isHalfCell() : false;
    }

    bool isSilicon(const DetId& id) const {
      return (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi ||
              (id.det() == DetId::Forward && id.subdetId() == static_cast<int>(ForwardSubdetector::HFNose)));
    }

    bool isScintillator(const DetId& id) const { return id.det() == DetId::HGCalHSc; }

    bool isScintillatorFine(const DetId& id) const {
      if (!isScintillator(id)) {
        return false;
      }
      const int32_t i = denseId(id);
      return (i >= 0) ? scint_[scintLocal(i)].isScintillatorFine() : false;
    }

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

    float getPhi(const DetId& id) const {
      const int32_t i = denseId(id);
      // atan2 gives the RecHitTools convention: +pi at the boundary where
      // GlobalPoint::phi wraps to -pi
      return (i >= 0) ? std::atan2(common_[i].y(), common_[i].x()) : 0.f;
    }

    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const {
      return hitEnergy / std::cosh(getEta(position, vertex_z));
    }

    float getPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const {
      return getPt(getPosition(id), hitEnergy, vertex_z);
    }

    int getScintMaxIphi(const DetId& id) const {
      if (!isScintillator(id)) {
        return 0;
      }
      const int32_t i = denseId(id);
      return (i >= 0) ? scint_[scintLocal(i)].scintMaxIphi() : 0;
    }

    unsigned int lastLayerEE(bool nose = false) const {
      return nose ? HFNoseDetId::HFNoseLayerEEmax : common_.lastLayerEE();
    }
    unsigned int lastLayerFH() const { return common_.lastLayerFH(); }
    unsigned int firstLayerBH() const { return common_.firstLayerBH(); }
    unsigned int lastLayerBH() const { return common_.lastLayerBH(); }
    unsigned int lastLayer(bool nose = false) const {
      return nose ? common_.noseLastLayer() : common_.numberOfLayers();
    }
    unsigned int getNumberOfLayers() const { return common_.numberOfLayers(); }
    unsigned int lastLayerECAL() const { return common_.lastLayerECAL(); }
    unsigned int lastLayerBarrel() const { return common_.lastLayerBarrel(); }
    unsigned int maxNumberOfWafersPerLayer(bool nose = false) const {
      return nose ? common_.maxNumberOfWafersNose() : common_.maxNumberOfWafersPerLayer();
    }
    int getScintMaxIphi() const { return common_.bhMaxIphi(); }
    int getGeometryType() const { return common_.geometryType(); }

    std::pair<uint32_t, uint32_t> firstAndLastLayer(DetId::Detector det, int subdet) const {
      if ((det == DetId::HGCalEE) || ((det == DetId::Forward) && (subdet == ForwardSubdetector::HGCEE))) {
        return {1, lastLayerEE()};
      } else if ((det == DetId::HGCalHSi) || ((det == DetId::Forward) && (subdet == ForwardSubdetector::HGCHEF))) {
        return {lastLayerEE() + 1, lastLayerFH()};
      } else if ((det == DetId::Forward) && (subdet == ForwardSubdetector::HFNose)) {
        return {1, common_.noseLastLayer()};
      } else {
        return {firstLayerBH(), lastLayerBH()};
      }
    }

    // corners must be the RecHitTools default of 3; the column is
    // precomputed at that value
    bool maskCell(const DetId& id, int corners = 3) const {
      const int32_t i = denseId(id);
      return (i >= 0) ? common_[i].masked() : false;
    }

    hgcal::RecHitTools::siliconWaferInfo getWaferInfo(const DetId& id) const {
      const int32_t i = denseId(id);
      if (i < 0 || !isSilicon(id)) {
        return {};
      }
      auto const si = silicon_[siliconLocal(i)];
      return {
          si.waferType(), si.waferPartialType(), si.waferOrientation(), si.waferPlacementIndex(), common_[i].cassette()};
    }

    hgcal::RecHitTools::scintillatorTileInfo getTileInfo(const DetId& id) const {
      const int32_t i = denseId(id);
      if (i < 0 || !isScintillator(id)) {
        return {};
      }
      auto const sc = scint_[scintLocal(i)];
      return {sc.tileType(), sc.tileSipm(), common_[i].cassette()};
    }

    TICLGeomCommonSoAConstView const& commonView() const { return common_; }
    TICLGeomSiliconSoAConstView const& siliconView() const { return silicon_; }
    TICLGeomScintSoAConstView const& scintView() const { return scint_; }
    TICLGeomLookupSoAConstView const& lookupView() const { return lookup_; }
    TICLGeomLayersSoAConstView const& layersView() const { return layers_; }

  private:
    TICLGeomCommonSoAConstView common_;
    TICLGeomSiliconSoAConstView silicon_;
    TICLGeomScintSoAConstView scint_;
    TICLGeomLookupSoAConstView lookup_;
    TICLGeomLayersSoAConstView layers_;
  };

}  // namespace ticlgeom

#endif  // RecoLocalCalo_HGCalRecAlgos_TICLGeomTools_h
