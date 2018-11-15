#ifndef Geometry_EcalMapping_EcalElectronicsMappingBuilder
#define Geometry_EcalMapping_EcalElectronicsMappingBuilder

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"

namespace edm {
  class ParameterSet;
}

class EcalElectronicsMappingBuilder : public edm::ESProducer
{
 public:
  EcalElectronicsMappingBuilder(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<EcalElectronicsMapping>;

  ReturnType produce(const EcalMappingRcd&);

 private:
  void FillFromDatabase(const std::vector<EcalMappingElement>& ee,
                        EcalElectronicsMapping& theMap);
};
#endif
