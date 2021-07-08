#ifndef __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__
#define __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__

#include <array>
#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class CaloGeometry;
class CaloSubdetectorGeometry;
class DetId;

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace hgcal {
  class RecHitTools {
  public:
    RecHitTools() : geom_(nullptr), fhOffset_(0), bhOffset_(0), fhLastLayer_(0), noseLastLayer_(0), geometryType_(0) {}
    ~RecHitTools() {}

    void setGeometry(CaloGeometry const&);
    const CaloSubdetectorGeometry* getSubdetectorGeometry(const DetId& id) const;

    GlobalPoint getPosition(const DetId& id) const;
    GlobalPoint getPositionLayer(int layer, bool nose = false) const;
    // zside returns +/- 1
    int zside(const DetId& id) const;

    std::float_t getSiThickness(const DetId&) const;
    std::float_t getRadiusToSide(const DetId&) const;
    int getSiThickIndex(const DetId&) const;

    std::pair<float, float> getScintDEtaDPhi(const DetId&) const;

    unsigned int getLayer(DetId::Detector type, bool nose = false) const;
    unsigned int getLayer(ForwardSubdetector type) const;
    unsigned int getLayer(const DetId&) const;
    unsigned int getLayerWithOffset(const DetId&) const;
    std::pair<int, int> getWafer(const DetId&) const;
    std::pair<int, int> getCell(const DetId&) const;

    bool isHalfCell(const DetId&) const;

    bool isSilicon(const DetId&) const;
    bool isScintillator(const DetId&) const;

    bool isOnlySilicon(const unsigned int layer) const;

    // 4-vector helper functions using GlobalPoint
    float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const;
    float getPhi(const GlobalPoint& position) const;
    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

    // 4-vector helper functions using DetId
    float getEta(const DetId& id, const float& vertex_z = 0.) const;
    float getPhi(const DetId& id) const;
    float getPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const;

    inline const CaloGeometry* getGeometry() const { return geom_; };
    unsigned int lastLayerEE(bool nose = false) const { return (nose ? HFNoseDetId::HFNoseLayerEEmax : fhOffset_); }
    unsigned int lastLayerFH() const { return fhLastLayer_; }
    unsigned int firstLayerBH() const { return bhOffset_ + 1; }
    unsigned int lastLayerBH() const { return bhLastLayer_; }
    unsigned int lastLayer(bool nose = false) const { return (nose ? noseLastLayer_ : bhLastLayer_); }
    unsigned int maxNumberOfWafersPerLayer(bool nose = false) const {
      return (nose ? maxNumberOfWafersNose_ : maxNumberOfWafersPerLayer_);
    }
    inline int getScintMaxIphi() const { return bhMaxIphi_; }
    inline int getGeometryType() const { return geometryType_; }
    bool maskCell(const DetId& id, int corners = 3) const;

  private:
    const CaloGeometry* geom_;
    unsigned int fhOffset_, bhOffset_, bhLastLayer_, fhLastLayer_, noseLastLayer_;
    unsigned int maxNumberOfWafersPerLayer_, maxNumberOfWafersNose_;
    int geometryType_;
    int bhMaxIphi_;
  };
}  // namespace hgcal

#endif
