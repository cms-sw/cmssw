#ifndef Geometry_TrackerNumberingBuilder_TrackerGeometricDetExtraESModule_H
#define Geometry_TrackerNumberingBuilder_TrackerGeometricDetExtraESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"

class  TrackerGeometricDetExtraESModule: public edm::ESProducer {

 public:
  TrackerGeometricDetExtraESModule(const edm::ParameterSet & p);

  std::unique_ptr<std::vector<GeometricDetExtra> > produce(const IdealGeometryRecord &);

 protected:

 private:
  void putOne(std::vector<GeometricDetExtra> & gde, const GeometricDet* gd, const DDExpandedView& ev, int lev );

  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geometricDetToken_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  edm::ESGetToken<PGeometricDetExtra, PGeometricDetExtraRcd> pgToken_;
  const bool fromDDD_;

};

#endif

