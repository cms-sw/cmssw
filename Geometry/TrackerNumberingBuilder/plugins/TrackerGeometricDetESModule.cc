#include "Geometry/TrackerNumberingBuilder/plugins/TrackerGeometricDetESModule.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "Geometry/Records/interface/PGeometricDetRcd.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "CondDBCmsTrackerConstruction.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

using namespace edm;

TrackerGeometricDetESModule::TrackerGeometricDetESModule(const edm::ParameterSet & p) 
  : fromDDD_(p.getParameter<bool>("fromDDD")) 
{
  setWhatProduced(this);
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule() {}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produce(const IdealGeometryRecord & iRecord){ 
  if(fromDDD_){

    edm::ESHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    
    DDDCmsTrackerContruction theDDDCmsTrackerContruction;
    return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>(theDDDCmsTrackerContruction.construct(&(*cpv))));

  }else{

    edm::ESHandle<PGeometricDet> pgd;
    iRecord.get( pgd );
    
    CondDBCmsTrackerConstruction cdbtc;
    return std::auto_ptr<GeometricDet> ( const_cast<GeometricDet*>(cdbtc.construct( *pgd )));
  }
}

void TrackerGeometricDetESModule::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
						 const edm::IOVSyncValue & iosv, 
						 edm::ValidityInterval & oValidity)
{
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetESModule);
