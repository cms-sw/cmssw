#ifndef Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H
#define Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"

class  TrackerGeometricDetESModule: public edm::ESProducer {
 public:
  TrackerGeometricDetESModule(const edm::ParameterSet & p);
  virtual ~TrackerGeometricDetESModule(); 
  std::auto_ptr<GeometricDet>       produce(const IdealGeometryRecord &);

 protected:

 private:
  bool fromDDD_;

};


#endif




