
#ifndef Geometry_EcalMapping_EcalElectronicsMappingBuilder
#define Geometry_EcalMapping_EcalElectronicsMappingBuilder

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

// #include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
// #include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"

#include <vector>

namespace edm {
  class ParameterSet;
}

//
// class decleration
//

// class EcalElectronicsMappingBuilder : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
class EcalElectronicsMappingBuilder : public edm::ESProducer 
{
 public:
  EcalElectronicsMappingBuilder(const edm::ParameterSet&);
  ~EcalElectronicsMappingBuilder() override;
  
  typedef std::unique_ptr<EcalElectronicsMapping> ReturnType;
  
  // ReturnType produce(const IdealGeometryRecord&);
  ReturnType produce(const EcalMappingRcd&);
  
 private:
  void FillFromDatabase(const std::vector<EcalMappingElement>& ee,
                        EcalElectronicsMapping& theMap);
  
  // void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  // ----------member data ---------------------------
};

#endif
