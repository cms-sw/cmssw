#ifndef Geometry_ForwardGeometry_CastorHardcodeGeometryLoader_H
#define Geometry_ForwardGeometry_CastorHardcodeGeometryLoader_H 1

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/CastorTopology.h"

class CaloCellGeometry;
class CaloSubdetectorGeometry;
class HcalCastorDetId;


class CastorHardcodeGeometryLoader {
public:
  CastorHardcodeGeometryLoader();
  explicit CastorHardcodeGeometryLoader(const CastorTopology& ht);
  virtual ~CastorHardcodeGeometryLoader() { delete theTopology ; };
  
  virtual std::auto_ptr<CaloSubdetectorGeometry> load(DetId::Detector det, int subdet);
  std::auto_ptr<CaloSubdetectorGeometry> load();
  
private:
  void init();
  void fill(HcalCastorDetId::Section section,CaloSubdetectorGeometry* cg);
  void makeCell( const HcalCastorDetId &  detId ,
		 CaloSubdetectorGeometry* geom   ) const;

      CastorTopology* theTopology;
      const CastorTopology* extTopology;


  float theEMSectiondX;
  float theEMSectiondY;
  float theEMSectiondZ;
  float theHADSectiondX;
  float theHADSectiondY;
  float theHADSectiondZ;
  
};

#endif
