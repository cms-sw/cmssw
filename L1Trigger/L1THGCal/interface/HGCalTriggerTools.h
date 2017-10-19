#ifndef __L1Trigger_L1THGCal_HGCalTriggerTools_h__
#define __L1Trigger_L1THGCal_HGCalTriggerTools_h__

#include <array>
#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

class HGCalTriggerGeometryBase;
class DetId;

namespace edm {
  class Event;
  class EventSetup;
}

  class HGCalTriggerTools {
  public:
  HGCalTriggerTools() : geom_(nullptr), fhOffset_(0), bhOffset_(0) {}
    ~HGCalTriggerTools() {}

    void setEventSetup(const edm::EventSetup&);
    GlobalPoint getPosition(const DetId& id) const;
    unsigned int getLayerWithOffset(const DetId&) const;
    // unsigned int getLayer(ForwardSubdetector type) const;
    unsigned int getLayer(const DetId&) const;

    // 4-vector helper functions using GlobalPoint
    float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const;
    float getPhi(const GlobalPoint& position) const;
    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

    // 4-vector helper functions using DetId
    float getEta(const DetId& id, const float& vertex_z = 0.) const;
    float getPhi(const DetId& id) const;
    float getPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const;

    inline const HGCalTriggerGeometryBase * getTriggerGeometry() const {return geom_;};
    unsigned int lastLayerEE() const {return fhOffset_;}
    unsigned int lastLayerFH() const {return bhOffset_;}

  private:
    const HGCalTriggerGeometryBase* geom_;
    unsigned int        fhOffset_, bhOffset_;
  };


#endif
