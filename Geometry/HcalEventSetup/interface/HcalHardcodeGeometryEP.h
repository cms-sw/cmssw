#ifndef GEOMETRY_HCALEVENTSETUP_HCALHARDCODEGEOMETRYEP_H
#define GEOMETRY_HCALEVENTSETUP_HCALHARDCODEGEOMETRYEP_H 1


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"

//
// class decleration
//

class HcalHardcodeGeometryEP : public edm::ESProducer 
{
   public:
      HcalHardcodeGeometryEP(const edm::ParameterSet&);
      virtual ~HcalHardcodeGeometryEP();

      typedef boost::shared_ptr<CaloSubdetectorGeometry> ReturnType;

      ReturnType produceIdeal(   const IdealGeometryRecord&);
      ReturnType produceAligned( const HcalGeometryRecord& );

      void       idealRecordCallBack( const IdealGeometryRecord& );

   private:
      // ----------member data ---------------------------

      HcalHardcodeGeometryLoader* m_loader ;

      bool m_applyAlignment ;
};



#endif
