#include "EcalElectronicsMappingBuilder.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

EcalElectronicsMappingBuilder::EcalElectronicsMappingBuilder(const edm::ParameterSet&):
  eeToken_{setWhatProduced(this).consumesFrom<EcalMappingElectronics, EcalMappingElectronicsRcd>(edm::ESInputTag{})}
{}

// ------------ method called to produce the data  ------------
EcalElectronicsMappingBuilder::ReturnType
EcalElectronicsMappingBuilder::produce(const EcalMappingRcd& iRecord)
{
   auto prod = std::make_unique<EcalElectronicsMapping>();

   const auto& item = iRecord.get(eeToken_);

   const std::vector<EcalMappingElement>& ee = item.endcapItems();
   FillFromDatabase(ee,*prod);
   return prod;
}

void EcalElectronicsMappingBuilder::FillFromDatabase(const std::vector<EcalMappingElement>& ee,
						     EcalElectronicsMapping& theMap)   
{
  for (unsigned int i=0; i < ee.size(); i++) 
    {
      if (ee[i].electronicsid == 0)
	continue;
      if (ee[i].triggerid == 0)
	continue;
      theMap.assign(EEDetId::unhashIndex(i).rawId(),ee[i].electronicsid,ee[i].triggerid);
    }
  return;
}
