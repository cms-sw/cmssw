#ifndef GEOMETRY_HCALEVENTSETUP_CALOTOWERHARDCODEGEOMETRYEP_H
#define GEOMETRY_HCALEVENTSETUP_CALOTOWERHARDCODEGEOMETRYEP_H 1

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloTowerGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"

class HcalRecNumberingRecord;
class IdealGeometryRecord;


class CaloTowerHardcodeGeometryEP : public edm::ESProducer {
public:
  CaloTowerHardcodeGeometryEP(const edm::ParameterSet&);
  ~CaloTowerHardcodeGeometryEP() override;

  using ReturnType = std::unique_ptr<CaloSubdetectorGeometry>;

  ReturnType produce(const CaloTowerGeometryRecord&);

private:
  // ----------member data ---------------------------
  CaloTowerHardcodeGeometryLoader* loader_;
};

#endif
