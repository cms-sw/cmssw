#ifndef GEOMETRY_HCALEVENTSETUP_CALOTOWERHARDCODEGEOMETRYEP_H
#define GEOMETRY_HCALEVENTSETUP_CALOTOWERHARDCODEGEOMETRYEP_H 1

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloTowerGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"

//
// class decleration
//
class HcalRecNumberingRecord;
class IdealGeometryRecord;


class CaloTowerHardcodeGeometryEP : public edm::ESProducer {
public:
  CaloTowerHardcodeGeometryEP(const edm::ParameterSet&);
  ~CaloTowerHardcodeGeometryEP() override;

  typedef std::unique_ptr<CaloSubdetectorGeometry> ReturnType;

  ReturnType produce(const CaloTowerGeometryRecord&);

  void       idealRecordCallBack( const HcalRecNumberingRecord& ) {}

private:
      // ----------member data ---------------------------
  CaloTowerHardcodeGeometryLoader* loader_;
};


#endif
