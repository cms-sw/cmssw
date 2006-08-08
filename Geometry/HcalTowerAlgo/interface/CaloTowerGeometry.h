#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H 1

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

/** \class CaloTowerGeometry
  *  
  * Only DetId::Calo, subdet=1 DetIds are handled by this class.
  *
  * $Date: 2005/10/06 01:02:04 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class CaloTowerGeometry : public CaloSubdetectorGeometry {
public:
  CaloTowerGeometry();
  virtual ~CaloTowerGeometry();  
};

#endif
