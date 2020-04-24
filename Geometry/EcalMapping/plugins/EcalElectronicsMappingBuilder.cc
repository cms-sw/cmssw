
// user include files
#include "EcalElectronicsMappingBuilder.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include <iostream>
#include <fstream>


#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"



using namespace edm;


//
// constructors and destructor
//
EcalElectronicsMappingBuilder::EcalElectronicsMappingBuilder(const edm::ParameterSet& iConfig) :
  Mapping_ ( nullptr )
{
  //the following line is needed to tell the framework what
  // data is being produced
  // setWhatProduced(this);
  setWhatProduced(this, (dependsOn (&EcalElectronicsMappingBuilder::DBCallback)) );
  //now do what ever other initialization is needed
}


EcalElectronicsMappingBuilder::~EcalElectronicsMappingBuilder()
{ 
}

//
// member functions
//

void EcalElectronicsMappingBuilder::DBCallback (const EcalMappingElectronicsRcd& fRecord)
{

  edm::ESHandle <EcalMappingElectronics> item;
  fRecord.get (item);
  Mapping_ = item.product () ;
}


// ------------ method called to produce the data  ------------
EcalElectronicsMappingBuilder::ReturnType
// EcalElectronicsMappingBuilder::produce(const IdealGeometryRecord& iRecord)
EcalElectronicsMappingBuilder::produce(const EcalMappingRcd& iRecord)
{
   using namespace edm::es;
   auto prod = std::make_unique<EcalElectronicsMapping>();
   const std::vector<EcalMappingElement>& ee = Mapping_ -> endcapItems();
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





