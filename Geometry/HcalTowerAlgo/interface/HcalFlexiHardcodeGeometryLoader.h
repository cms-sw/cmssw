#ifndef GEOMETRY_HCALTOWERALGO_HCALFLEXIHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_HCALFLEXIHARDCODEGEOMETRYLOADER_H 1

/** \class HcalFlexiHardcodeGeometryLoader
 *
 * $Date: 2008/08/30 20:18:54 $
 * $Revision: 1.2 $
 * \author F.Ratnikov, UMd
*/

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"

class HcalTopology;

class HcalFlexiHardcodeGeometryLoader 
{
   public:

      HcalFlexiHardcodeGeometryLoader();
  
      CaloSubdetectorGeometry* load(const HcalTopology& fTopology);
  
};

#endif
