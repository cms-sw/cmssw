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
 * $Date: 2006/10/19 02:16:57 $
 * $Revision: 1.0 $
 * \author S. Banerjee
*/

class HcalDDDGeometryLoader : public CaloVGeometryLoader {

public:

  explicit HcalDDDGeometryLoader(const DDCompactView & cpv);
  virtual ~HcalDDDGeometryLoader();
  
  virtual std::auto_ptr<CaloSubdetectorGeometry> load(DetId::Detector , int );
  /// Load all of HCAL
  std::auto_ptr<CaloSubdetectorGeometry> load();
  
private:

  HcalDDDGeometryLoader();

  /// helper functions to make all the ids and cells, and put them into the
  /// vectors and mpas passed in.
  void fill(HcalSubdetector, HcalDDDGeometry*, CaloSubdetectorGeometry*);
  
  const CaloCellGeometry * makeCell(const HcalDetId &, 
				    HcalCellType::HcalCellType, double, 
				    double) const;
  
  HcalNumberingFromDDD* numberingFromDDD;

};

#endif
