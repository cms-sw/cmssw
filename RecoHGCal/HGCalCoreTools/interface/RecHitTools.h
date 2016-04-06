#ifndef __RecoHGCal_HGCalCoreTools_RecHitTools_h__
#define __RecoHGCal_HGCalCoreTools_RecHitTools_h__

#include <cmath>

class HGCalGeometry;
class HGCalDDDConstants;
class DetId;

namespace edm {
  class Event;
  class EventSetup;
}

namespace hgcal {
  class RecHitTools {    
  public:
    RecHitTools() : geom_(nullptr), ddd_(nullptr) {}
    ~RecHitTools() {}

    void getEvent(const edm::Event&);
    void getEventSetup(const edm::EventSetup&);
    
    std::float_t getSiThickness(const DetId&) const;
    std::float_t getRadiusToSide(const DetId&) const;

    bool isHalfCell(const DetId&) const;

  private:
    const HGCalGeometry* geom_;
    const HGCalDDDConstants* ddd_;
  };
}

#endif
