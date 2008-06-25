#ifndef Geometry_ForwardGeometry_CastorGeometry_h
#define Geometry_ForwardGeometry_CastorGeometry_h 1

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorTopology.h"

#include <vector>

class CastorGeometry : public CaloSubdetectorGeometry {
public:

  explicit CastorGeometry(const CastorTopology * topology);
  virtual ~CastorGeometry();
  
  virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det    = DetId::Detector ( 0 ) ,
						    int             subdet = 0 ) const ;

  virtual DetId getClosestCell(const GlobalPoint& r) const ;

private:
  const CastorTopology * theTopology;
  mutable DetId::Detector lastReqDet_;
  mutable int lastReqSubdet_;
  mutable std::vector<DetId> m_validIds;
};


#endif
