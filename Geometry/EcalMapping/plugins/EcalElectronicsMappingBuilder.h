
#ifndef Geometry_EcalMapping_EcalElectronicsMappingBuilder
#define Geometry_EcalMapping_EcalElectronicsMappingBuilder

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"


//
// class decleration
//

class EcalElectronicsMappingBuilder : public edm::ESProducer {
   public:
  EcalElectronicsMappingBuilder(const edm::ParameterSet&);
  ~EcalElectronicsMappingBuilder();

  typedef std::auto_ptr<EcalElectronicsMapping> ReturnType;

  // ReturnType produce(const IdealGeometryRecord&);
  ReturnType produce(const EcalMappingRcd&);

private:
  void parseTextMap(const std::string& filename,EcalElectronicsMapping& theMap);
  std::string mapFile_;
      // ----------member data ---------------------------
};

#endif
