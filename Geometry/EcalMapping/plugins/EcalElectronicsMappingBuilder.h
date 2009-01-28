
#ifndef Geometry_EcalMapping_EcalElectronicsMappingBuilder
#define Geometry_EcalMapping_EcalElectronicsMappingBuilder

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


// class EcalMappingRcd;


// #include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"

#include <vector>

//
// class decleration
//

// class EcalElectronicsMappingBuilder : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
class EcalElectronicsMappingBuilder : public edm::ESProducer 
{
 public:
  EcalElectronicsMappingBuilder(const edm::ParameterSet&);
  ~EcalElectronicsMappingBuilder();
  
  typedef std::auto_ptr<EcalElectronicsMapping> ReturnType;
  
  // ReturnType produce(const IdealGeometryRecord&);
  ReturnType produce(const EcalMappingRcd&);
  
  void DBCallback (const EcalMappingElectronicsRcd& fRecord);
  
 private:
  void FillFromDatabase(const std::vector<EcalMappingElement>& ee,
                        EcalElectronicsMapping& theMap);
  
  
  const EcalMappingElectronics* Mapping_ ;
  // void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  // ----------member data ---------------------------
};

#endif
