#ifndef HcalGeometry_h
#define HcalGeometry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

class HcalGeometry : public CaloSubdetectorGeometry {
public:

  HcalGeometry();
  /// The HcalGeometry will delete all its cell geometries at destruction time
  virtual ~HcalGeometry();
  
  virtual std::vector<DetId> getValidDetIds(DetId::Detector det, int subdet) const;
  /// TODO: add "nearest cell" code eventually

private:
  mutable DetId::Detector lastReqDet_;
  mutable int lastReqSubdet_;
};


#endif

