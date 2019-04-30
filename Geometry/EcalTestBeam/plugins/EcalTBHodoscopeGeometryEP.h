#ifndef GEOMETRY_ECALEVENTSETUP_ECALTBHODOSCOPEGEOMETRYEPEP_H
#define GEOMETRY_ECALEVENTSETUP_ECALTBHODOSCOPEGEOMETRYEPEP_H 1


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/EcalTestBeam/plugins/EcalTBHodoscopeGeometryLoaderFromDDD.h"

//
// class declaration
//

class EcalTBHodoscopeGeometryEP : public edm::ESProducer {
 public:
  EcalTBHodoscopeGeometryEP(const edm::ParameterSet&);
  ~EcalTBHodoscopeGeometryEP() override = default;
  
  typedef std::unique_ptr<CaloSubdetectorGeometry> ReturnType;
  
  ReturnType produce(const IdealGeometryRecord&);

 private:

  // ----------member data ---------------------------
  EcalTBHodoscopeGeometryLoaderFromDDD loader_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
};



#endif
