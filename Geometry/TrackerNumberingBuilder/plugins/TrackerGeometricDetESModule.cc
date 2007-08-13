#include "Geometry/TrackerNumberingBuilder/plugins/TrackerGeometricDetESModule.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
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
  if ( fromDDD_ ) {
    setWhatProduced(this, &TrackerGeometricDetESModule::produceFromDDDXML);
    findingRecord<IdealGeometryRecord>();
  } else {
    setWhatProduced(this, &TrackerGeometricDetESModule::produceFromPGeometricDet);
    findingRecord<PGeometricDetRcd>();
  }
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule() {}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produceFromDDDXML(const IdealGeometryRecord & iRecord){ 
  //
  // get the DDCompactView first
  //
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get( cpv );
  
  DDDCmsTrackerContruction theDDDCmsTrackerContruction;
  return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>(theDDDCmsTrackerContruction.construct(&(*cpv))));
}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produceFromPGeometricDet(const PGeometricDetRcd & iRecord){ 
  edm::ESHandle<PGeometricDet> pgd;
  iRecord.get( pgd );
  
  CondDBCmsTrackerConstruction cdbtc;
  //  std::auto_ptr<GeometricDet> tt (
  return std::auto_ptr<GeometricDet> ( const_cast<GeometricDet*>(cdbtc.construct( *pgd )));

  //  DDDCmsTrackerConstruction theDDDCmsTrackerContruction;
  //  return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>(theDDDCmsTrackerContruction.construct(&(*cpv))));
}


void TrackerGeometricDetESModule::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
						 const edm::IOVSyncValue & iosv, 
						 edm::ValidityInterval & oValidity)
{
  // TO CHECK: can we get the iov from the PoolDBESSource?  if not, why not?
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetESModule);
