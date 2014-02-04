#ifndef Geometry_HcalTowerAlgo_HcalDDDGeometryLoader_H
#define Geometry_HcalTowerAlgo_HcalDDDGeometryLoader_H 1

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class DDCompactView;
class CaloCellGeometry;
class HcalDetId;

/** \class HcalDDDGeometryLoader
 *
 *
 * \note The Geometry as loaded from DDD
 *   
 * $Date: 2012/10/29 07:30:41 $
 * $Revision: 1.7 $
 * \author S. Banerjee
*/

class HcalDDDGeometryLoader {

public:

  explicit HcalDDDGeometryLoader(const DDCompactView & cpv);
  virtual ~HcalDDDGeometryLoader();
  
  typedef CaloSubdetectorGeometry* ReturnType ;
  ReturnType load(const HcalTopology& topo, DetId::Detector , int );
  /// Load all of HCAL
  ReturnType load(const HcalTopology& topo);
  
private:

  HcalDDDGeometryLoader();

  /// helper functions to make all the ids and cells, and put them into the
  /// vectors and mpas passed in.
  void fill(HcalSubdetector, HcalDDDGeometry*, CaloSubdetectorGeometry*);
  
  void makeCell( const HcalDetId &, 
		 HcalCellType, double, 
		 double, CaloSubdetectorGeometry* geom) const;
  
  HcalDDDSimConstants* hcalConstants;

  HcalTopology* dummyTopology_;

};

#endif
