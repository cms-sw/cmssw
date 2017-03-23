#ifndef __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__
#define __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__

#include <array>
#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class CaloGeometry;
class DetId;

namespace edm {
  class Event;
  class EventSetup;
}

namespace hgcal {
  class RecHitTools {
  public:
  RecHitTools() : geom_(nullptr) {}
    ~RecHitTools() {}

    void getEvent(const edm::Event&);
    void getEventSetup(const edm::EventSetup&);

    GlobalPoint getPosition(const DetId& id) const;
    
    std::float_t getSiThickness(const DetId&) const;
    std::float_t getRadiusToSide(const DetId&) const;

    unsigned int getLayer(const DetId&) const;
    unsigned int getLayerWithOffset(const DetId&) const;
    unsigned int getWafer(const DetId&) const;
    unsigned int getCell(const DetId&) const;

    bool isHalfCell(const DetId&) const;

    // 4-vector helper functions using GlobalPoint
    float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const;
    float getPhi(const GlobalPoint& position) const;
    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

    // 4-vector helper functions using DetId
    float getEta(const DetId& id, const float& vertex_z = 0.) const;
    float getPhi(const DetId& id) const;
    float getPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const;

    inline const CaloGeometry * getGeometry() const {return geom_;};
  private:
    const CaloGeometry* geom_;
  };
}

#endif
