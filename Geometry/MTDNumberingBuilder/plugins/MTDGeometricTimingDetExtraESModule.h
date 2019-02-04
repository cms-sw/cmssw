#ifndef Geometry_MTDNumberingBuilder_MTDGeometricTimingDetExtraESModule_H
#define Geometry_MTDNumberingBuilder_MTDGeometricTimingDetExtraESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDetExtra.h"

class  MTDGeometricTimingDetExtraESModule: public edm::ESProducer {

 public:
  MTDGeometricTimingDetExtraESModule(const edm::ParameterSet & p);
  ~MTDGeometricTimingDetExtraESModule() override; 
  std::unique_ptr<std::vector<GeometricTimingDetExtra> > produce(const IdealGeometryRecord &);

 protected:

 private:
  void putOne(std::vector<GeometricTimingDetExtra> & gde, const GeometricTimingDet* gd, const DDExpandedView& ev, int lev );

  bool fromDDD_;

};


#endif




