#ifndef CaloVGeometryLoader_h
#define CaloVGeometryLoader_h

#include "DataFormats/DetId/interface/DetId.h"
#include <memory>

class CaloSubdetectorGeometry;

class CaloVGeometryLoader {
public:
  virtual std::auto_ptr<CaloSubdetectorGeometry> load(DetId::Detector det, int subdet) = 0;
};

#endif

