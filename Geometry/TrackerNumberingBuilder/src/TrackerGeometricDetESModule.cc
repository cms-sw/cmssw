#include "Geometry/TrackerNumberingBuilder/interface/TrackerGeometricDetESModule.h"
#include "Geometry/TrackerNumberingBuilder/interface/DDDCmsTrackerContruction.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

using namespace edm;

TrackerGeometricDetESModule::TrackerGeometricDetESModule(const edm::ParameterSet & p) 
{
    setWhatProduced(this);
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule() {}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produce(const IdealGeometryRecord & iRecord){ 
  //
  // get the DDCompactView first
  //
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get( cpv );
  
  DDDCmsTrackerContruction theDDDCmsTrackerContruction;
  return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>(theDDDCmsTrackerContruction.construct(&(*cpv))));
}


DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetESModule)
