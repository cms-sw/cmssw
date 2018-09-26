#ifndef Geometry_HcalEventSetup_HcalHardcodeGeometryEP_H
#define Geometry_HcalEventSetup_HcalHardcodeGeometryEP_H 1

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CaloSubdetectorGeometry;
class HcalRecNumberingRecord;
class HcalGeometryRecord;

class HcalHardcodeGeometryEP : public edm::ESProducer {

public:
  HcalHardcodeGeometryEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<CaloSubdetectorGeometry>;

  ReturnType produceIdeal(const HcalRecNumberingRecord&);
  ReturnType produceAligned(const HcalGeometryRecord& );

private:

  bool              useOld_;
};
#endif
