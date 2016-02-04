#ifndef Geometry_HcalEventSetup_HcalDDDGeometryEP_H
#define Geometry_HcalEventSetup_HcalDDDGeometryEP_H 1


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometryLoader.h"

//
// class decleration
//

class HcalDDDGeometryEP : public edm::ESProducer 
{
   public:

      HcalDDDGeometryEP(const edm::ParameterSet&);
      ~HcalDDDGeometryEP();

      typedef boost::shared_ptr<CaloSubdetectorGeometry> ReturnType;
  
      void idealRecordCallBack(const IdealGeometryRecord&);

      ReturnType produceIdeal(const IdealGeometryRecord&);
      ReturnType produceAligned(const HcalGeometryRecord&);

private:

  // ----------member data ---------------------------

      HcalDDDGeometryLoader* m_loader ;

      const DDCompactView* m_cpv ;

      bool m_applyAlignment ;
};

#endif
