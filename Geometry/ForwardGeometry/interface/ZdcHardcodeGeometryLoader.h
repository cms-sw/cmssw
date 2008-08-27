#ifndef Geometry_ForwardGeometry_ZdcHardcodeGeometryLoader_H
#define Geometry_ForwardGeometry_ZdcHardcodeGeometryLoader_H 1

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include <boost/shared_ptr.hpp>

class CaloCellGeometry;
class CaloSubdetectorGeometry;
class HcalZDCDetId;

/** \class ZdcHardcodeGeometryLoader
 *
 * $Date: 2008/04/21 22:20:16 $
 * $Revision: 1.3 $
 * \author E. Garcia - UIC
*/

class ZdcHardcodeGeometryLoader {
public:

  typedef CaloSubdetectorGeometry* ReturnType ;

  ZdcHardcodeGeometryLoader();
  explicit ZdcHardcodeGeometryLoader(const ZdcTopology& ht);
  virtual ~ZdcHardcodeGeometryLoader() { delete theTopology ; }
  
  virtual ReturnType load(DetId::Detector det, int subdet);
  ReturnType load();
  
private:
  void init();
  void fill(HcalZDCDetId::Section section,CaloSubdetectorGeometry* cg);
  CaloCellGeometry* makeCell(const HcalZDCDetId & detId,
			     CaloSubdetectorGeometry* geom) const;

  ZdcTopology*       theTopology;
  const ZdcTopology* extTopology;
      
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
