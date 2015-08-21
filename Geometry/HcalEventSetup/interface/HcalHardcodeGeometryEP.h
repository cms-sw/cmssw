#ifndef Geometry_HcalEventSetup_HcalHardcodeGeometryEP_H
#define Geometry_HcalEventSetup_HcalHardcodeGeometryEP_H 1

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class declarations
class CaloSubdetectorGeometry;
class HcalRecNumberingRecord;
class HcalGeometryRecord;

class HcalHardcodeGeometryEP : public edm::ESProducer {

public:
  HcalHardcodeGeometryEP(const edm::ParameterSet&);
  virtual ~HcalHardcodeGeometryEP();

  typedef boost::shared_ptr<CaloSubdetectorGeometry> ReturnType;

  ReturnType produceIdeal(const HcalRecNumberingRecord&);
  ReturnType produceAligned(const HcalGeometryRecord& );

  void       idealRecordCallBack(const HcalRecNumberingRecord&) {}

private:
  edm::ParameterSet ps0;
  bool              useOld_;
};
#endif
