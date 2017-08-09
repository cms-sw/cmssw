#ifndef CaloVGeometryLoader_h
#define CaloVGeometryLoader_h

#include "DataFormats/DetId/interface/DetId.h"
#include <memory>

class CaloSubdetectorGeometry;

/** \class CaloVGeometryLoader 

  Abstract base class for a subdetector geometry loader.
 */
class CaloVGeometryLoader 
{
   public:
      virtual ~CaloVGeometryLoader() = default;
      /// Load the subdetector geometry for the specified det and subdet
      virtual std::unique_ptr<CaloSubdetectorGeometry> 
                     load( DetId::Detector det, int subdet ) = 0;
};

#endif

