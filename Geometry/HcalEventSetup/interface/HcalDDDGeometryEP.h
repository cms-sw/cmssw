#ifndef Geometry_HcalEventSetup_HcalDDDGeometryEP_H
#define Geometry_HcalEventSetup_HcalDDDGeometryEP_H 1

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometryLoader.h"

class HcalDDDGeometryEP : public edm::ESProducer {

public:

  HcalDDDGeometryEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<CaloSubdetectorGeometry>;

  ReturnType produceIdeal(const HcalRecNumberingRecord&);
  ReturnType produceAligned(const HcalGeometryRecord&);

private:

  bool m_applyAlignment ;
};
#endif
