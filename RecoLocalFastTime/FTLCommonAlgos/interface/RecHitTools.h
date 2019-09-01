#ifndef __RecoLocalFastTime_FTLCommonAlgos_RecHitTools_h__
#define __RecoLocalFastTime_FTLCommonAlgos_RecHitTools_h__

#include <array>
#include <cmath>
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace ftl {
  class RecHitTools {
  public:
    RecHitTools() : geom_(nullptr), ddd_(nullptr) {}
    ~RecHitTools() {}

    enum HitType { UNKNOWN = 0, LYSO = 1, Silicon = 2 };

    void getEvent(const edm::Event&);
    void getEventSetup(const edm::EventSetup&);

    GlobalPoint getPosition(const DetId& id) const;
    FlatTrd::CornersVec getCorners(const DetId& id) const;

    HitType getHitType(const DetId& id) const;

  private:
    const FastTimeGeometry* geom_;
    const FastTimeDDDConstants* ddd_;
  };
}  // namespace ftl

#endif
