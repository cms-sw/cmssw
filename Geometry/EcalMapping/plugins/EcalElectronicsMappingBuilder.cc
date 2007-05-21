
// user include files
#include "EcalElectronicsMappingBuilder.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include <iostream>
#include <fstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
EcalElectronicsMappingBuilder::EcalElectronicsMappingBuilder(const edm::ParameterSet& iConfig) :
  mapFile_(iConfig.getUntrackedParameter<std::string>("MapFile",""))
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  
  //now do what ever other initialization is needed
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
   using namespace edm::es;
   std::auto_ptr<EcalElectronicsMapping> prod(new EcalElectronicsMapping());

   if (!mapFile_.empty()) {
     parseTextMap(mapFile_,*prod);
   }
   return prod;
}

void EcalElectronicsMappingBuilder::parseTextMap(const std::string& filename, EcalElectronicsMapping& theMap) {
  edm::FileInPath eff(filename);
  
  std::ifstream f(eff.fullPath().c_str());
  if (!f.good())
    return; 
  
 // uint32_t detid, elecid, triggerid;

 int ix, iy, iz, CL;
 // int dccid, towerid, stripid, xtalid;
 // int tccid, tower, ipseudostrip, xtalinps;
 int dccid, towerid, pseudostrip_in_SC, xtal_in_pseudostrip;
 int tccid, tower, pseudostrip_in_TCC, pseudostrip_in_TT;

 while ( ! f.eof()) {
	// f >> detid >> elecid >> triggerid; 
	f >> ix >> iy >> iz >> CL >> dccid >> towerid >> pseudostrip_in_SC >> xtal_in_pseudostrip >> tccid >> tower >> 
		pseudostrip_in_TCC >> pseudostrip_in_TT ;

	EEDetId detid(ix,iy,iz,EEDetId::XYMODE);
	// std::cout << " dcc tower ps_in_SC xtal_in_ps " << dccid << " " << towerid << " " << pseudostrip_in_SC << " " << xtal_in_pseudostrip << std::endl;
	EcalElectronicsId elecid(dccid,towerid, pseudostrip_in_SC, xtal_in_pseudostrip);
        // std::cout << " tcc tt ps_in_TT xtal_in_ps " << tccid << " " << tower << " " << pseudostrip_in_TT << " " << xtal_in_pseudostrip << std::endl;
	EcalTriggerElectronicsId triggerid(tccid, tower, pseudostrip_in_TT, xtal_in_pseudostrip);
 
	theMap.assign(detid,elecid,triggerid);
 }

  f.close();
  return;
}
