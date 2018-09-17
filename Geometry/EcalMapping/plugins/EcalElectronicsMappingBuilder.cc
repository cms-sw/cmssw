
// user include files
#include "EcalElectronicsMappingBuilder.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

//
// constructors and destructor
//
EcalElectronicsMappingBuilder::EcalElectronicsMappingBuilder(const edm::ParameterSet&)
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
}


EcalElectronicsMappingBuilder::~EcalElectronicsMappingBuilder()
{ 
}

//
// member functions
//

// ------------ method called to produce the data  ------------
EcalElectronicsMappingBuilder::ReturnType
// EcalElectronicsMappingBuilder::produce(const IdealGeometryRecord& iRecord)
EcalElectronicsMappingBuilder::produce(const EcalMappingRcd& iRecord)
{
   auto prod = std::make_unique<EcalElectronicsMapping>();

   const EcalMappingElectronicsRcd& fRecord = iRecord.getRecord<EcalMappingElectronicsRcd>();
   edm::ESHandle <EcalMappingElectronics> item;
   fRecord.get(item);

   const std::vector<EcalMappingElement>& ee = item->endcapItems();
   FillFromDatabase(ee,*prod);
   return prod;
}

void EcalElectronicsMappingBuilder::FillFromDatabase(const std::vector<EcalMappingElement>& ee,
						     EcalElectronicsMapping& theMap)   
{
  //  std::cout << " --- Reading the EE mapping from Database --- " << std::endl;
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
