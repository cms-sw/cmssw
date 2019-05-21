#ifndef GEOMETRY_HCALEVENTSETUP_CALOTOWERHARDCODEGEOMETRYEP_H
#define GEOMETRY_HCALEVENTSETUP_CALOTOWERHARDCODEGEOMETRYEP_H 1

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/Records/interface/CaloTowerGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"

class HcalRecNumberingRecord;
class IdealGeometryRecord;


class CaloTowerHardcodeGeometryEP : public edm::ESProducer {
public:
  CaloTowerHardcodeGeometryEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<CaloSubdetectorGeometry>;

  ReturnType produce(const CaloTowerGeometryRecord&);

private:
  // ----------member data ---------------------------
  CaloTowerHardcodeGeometryLoader loader_;
  edm::ESGetToken<CaloTowerTopology, HcalRecNumberingRecord> cttopoToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcaltopoToken_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> consToken_;
};

#endif
