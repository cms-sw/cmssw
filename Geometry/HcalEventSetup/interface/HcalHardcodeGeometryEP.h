#ifndef Geometry_HcalEventSetup_HcalHardcodeGeometryEP_H
#define Geometry_HcalEventSetup_HcalHardcodeGeometryEP_H 1

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

class CaloSubdetectorGeometry;
class HcalRecNumberingRecord;
class HcalGeometryRecord;
class HcalDDDRecConstants;
class HcalTopology;

class HcalHardcodeGeometryEP : public edm::ESProducer {
public:
  HcalHardcodeGeometryEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<CaloSubdetectorGeometry>;

  ReturnType produceAligned(const HcalGeometryRecord&);

private:
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> consToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topologyToken_;
  bool useOld_;
};
#endif
