#ifndef __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__
#define __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__

#include <array>
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
  RecHitTools() : geom_({ {nullptr,nullptr} }), ddd_({ {nullptr,nullptr} }) {}
    ~RecHitTools() {}

    void getEvent(const edm::Event&);
    void getEventSetup(const edm::EventSetup&);
    
    std::float_t getSiThickness(const DetId&) const;
    std::float_t getRadiusToSide(const DetId&) const;
    
    bool isHalfCell(const DetId&) const;

  private:
    std::array<const HGCalGeometry*,2>     geom_;
    std::array<const HGCalDDDConstants*,2> ddd_;
  };
}

#endif
