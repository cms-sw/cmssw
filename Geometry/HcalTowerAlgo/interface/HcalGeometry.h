#ifndef HcalGeometry_h
#define HcalGeometry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class HcalGeometry : public CaloSubdetectorGeometry {
public:

  explicit HcalGeometry(const HcalTopology * topology);
  /// The HcalGeometry will delete all its cell geometries at destruction time
  virtual ~HcalGeometry();
  
  /// overriden to deal with detid representation issues
  virtual bool present(const DetId& id) const;
  /// overriden to deal with detid representation issues
  virtual const CaloCellGeometry* getGeometry(const DetId& id) const;
  

  virtual std::vector<DetId> getValidDetIds(DetId::Detector det, int subdet) const;
  virtual const DetId getClosestCell(const GlobalPoint& r) const ;

private:
  /// helper methods for getClosestCell
  int etaRing(HcalSubdetector bc, double abseta) const;
  int phiBin(double phi, int etaring) const;

  const HcalTopology * theTopology;
  mutable DetId::Detector lastReqDet_;
  mutable int lastReqSubdet_;
};


#endif

