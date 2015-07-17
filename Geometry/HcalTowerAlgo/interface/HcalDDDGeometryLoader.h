#ifndef Geometry_HcalTowerAlgo_HcalDDDGeometryLoader_H
#define Geometry_HcalTowerAlgo_HcalDDDGeometryLoader_H 1

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class CaloCellGeometry;
class HcalDetId;

/** \class HcalDDDGeometryLoader
 *
 *
 * \note The Geometry as loaded from DDD
 *   
 * \author S. Banerjee
*/

class HcalDDDGeometryLoader {

public:

  explicit HcalDDDGeometryLoader(const HcalDDDRecConstants * hcons);
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
		 const HcalCellType& , double, 
		 double, CaloSubdetectorGeometry* geom) const;
  
  const HcalDDDRecConstants* hcalConstants;

  HcalTopology* dummyTopology_;

};

#endif
