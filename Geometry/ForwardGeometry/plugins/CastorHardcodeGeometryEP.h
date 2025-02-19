#ifndef Geometry_ForwardGeometry_CastorHardcodeGeometryEP_H
#define Geometry_ForwardGeometry_CastorHardcodeGeometryEP_H 1

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CastorGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorHardcodeGeometryLoader.h"

//
// class declaration
//

class CastorHardcodeGeometryEP : public edm::ESProducer {
   public:
      CastorHardcodeGeometryEP(const edm::ParameterSet&);
      ~CastorHardcodeGeometryEP();

      typedef std::auto_ptr<CaloSubdetectorGeometry> ReturnType;

      ReturnType produce(const CastorGeometryRecord&);
private:
      // ----------member data ---------------------------
  CastorHardcodeGeometryLoader* loader_;
};



#endif
