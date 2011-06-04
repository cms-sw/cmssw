#ifndef Geometry_HcalTowerAlgo_HcalDDDGeometryLoader_H
#define Geometry_HcalTowerAlgo_HcalDDDGeometryLoader_H 1

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometry.h"

class DDCompactView;
class CaloCellGeometry;
class HcalDetId;

/** \class HcalDDDGeometryLoader
 *
 *
 * \note The Geometry as loaded from DDD
 *   
 * $Date: 2010/04/12 23:35:37 $
 * $Revision: 1.5 $
 * \author S. Banerjee
*/

class HcalDDDGeometryLoader // : public CaloVGeometryLoader {
{
   public:

      explicit HcalDDDGeometryLoader(const DDCompactView & cpv);
      virtual ~HcalDDDGeometryLoader();
  
      typedef CaloSubdetectorGeometry* ReturnType ;
      ReturnType load(DetId::Detector , int );
      /// Load all of HCAL
      ReturnType load();
  
private:

  HcalDDDGeometryLoader();

  /// helper functions to make all the ids and cells, and put them into the
  /// vectors and mpas passed in.
  void fill(HcalSubdetector, HcalDDDGeometry*, CaloSubdetectorGeometry*);
  
  void makeCell( const HcalDetId &, 
		 HcalCellType, double, 
		 double, CaloSubdetectorGeometry* geom) const;
  
  HcalNumberingFromDDD* numberingFromDDD;

};

#endif
