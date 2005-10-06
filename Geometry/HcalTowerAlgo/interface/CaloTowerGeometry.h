#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H 1

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

/** \class CaloTowerGeometry
  *  
  * Only DetId::Calo, subdet=1 DetIds are handled by this class.
  *
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class CaloTowerGeometry : public CaloSubdetectorGeometry {
public:
  CaloTowerGeometry();
  virtual ~CaloTowerGeometry();  
};

#endif
