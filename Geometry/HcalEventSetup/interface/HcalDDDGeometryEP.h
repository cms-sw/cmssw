#ifndef Geometry_HcalEventSetup_HcalDDDGeometryEP_H
#define Geometry_HcalEventSetup_HcalDDDGeometryEP_H 1


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometryLoader.h"

//
// class decleration
//

class HcalDDDGeometryEP : public edm::ESProducer {

public:

  HcalDDDGeometryEP(const edm::ParameterSet&);
  ~HcalDDDGeometryEP();

  typedef std::auto_ptr<CaloSubdetectorGeometry> ReturnType;
  
  ReturnType produce(const IdealGeometryRecord&);

private:

  // ----------member data ---------------------------
  HcalDDDGeometryLoader* loader_;
};

#endif
