#ifndef Geometry_ForwardGeometry_ZdcHardcodeGeometryLoader_H
#define Geometry_ForwardGeometry_ZdcHardcodeGeometryLoader_H 1

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"

class CaloCellGeometry;
class CaloSubdetectorGeometry;
class HcalZDCDetId;

/** \class ZdcHardcodeGeometryLoader
 *
 * $Date: 2007/08/28 18:10:10 $
 * $Revision: 1.1 $
 * \author E. Garcia - UIC
*/

class ZdcHardcodeGeometryLoader {
public:
  ZdcHardcodeGeometryLoader();
  explicit ZdcHardcodeGeometryLoader(const ZdcTopology& ht);
  virtual ~ZdcHardcodeGeometryLoader() {};
  
  virtual std::auto_ptr<CaloSubdetectorGeometry> load(DetId::Detector det, int subdet);
  std::auto_ptr<CaloSubdetectorGeometry> load();
  
private:
  void init();
  void fill(HcalZDCDetId::Section section,CaloSubdetectorGeometry* cg);
  const CaloCellGeometry * makeCell(const HcalZDCDetId & detId,
				    CaloSubdetectorGeometry* geom) const;

  ZdcTopology theTopology;


  float theEMSectiondX;
  float theEMSectiondY;
  float theEMSectiondZ;
  float theLUMSectiondX;
  float theLUMSectiondY;
  float theLUMSectiondZ;
  float theHADSectiondX;
  float theHADSectiondY;
  float theHADSectiondZ;
  
};

#endif
