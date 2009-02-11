#include "L1TriggerConfig/L1ScalesProducers/interface/L1CaloInputScalesGenerator.h"

// system include files
#include <memory>
#include <iostream>
using std::cout;
using std::endl;
#include <iomanip>
using std::setprecision;
#include <fstream>
using std::ofstream;


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1CaloInputScalesGenerator::L1CaloInputScalesGenerator(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


L1CaloInputScalesGenerator::~L1CaloInputScalesGenerator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1CaloInputScalesGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

   ESHandle<CaloTPGTranscoder> caloTPGTranscoder;
   iSetup.get<CaloTPGRecord>().get(caloTPGTranscoder);
   
   EcalTPGScale* ecalTPGScale = new EcalTPGScale();
   ecalTPGScale->setEventSetup(iSetup);

   double output; 
   ofstream scalesFile("L1CaloInputScales.cfi");


   // Write the ecal scales, positive eta

   scalesFile << "es_module L1CaloInputScalesProducer = "
	      << "L1CaloInputScalesProducer {\n\tvdouble "
	      << "L1EcalEtThresholdsPositiveEta = {" << endl; 

   // loop over ietas, barrel
   for (unsigned short absIeta = 1; absIeta <= 28; absIeta++)
     {
       EcalSubdetector subdet = ( absIeta <= 17 ) ? EcalBarrel : EcalEndcap ;
       // 8 bits of input energy
       for (unsigned short input = 0; input <= 0xFF; input++)
	 {
	   output = ecalTPGScale->getTPGInGeV( (uint) input, 
					      EcalTrigTowerDetId(1, subdet,
								 absIeta, 1));
	   scalesFile << setprecision (8) << output;
	   if (absIeta == 28 && input == 0xFF)
	     {
	       scalesFile << "}";
	     }
	   else
	     {
	       scalesFile << ", ";
	     }
	 }
       scalesFile << endl;
     }

   // Write the ecal scales, negative eta

   scalesFile << endl << "\tvdouble L1EcalEtThresholdsNegativeEta = {" << endl;

   // loop over ietas, barrel first
   for (unsigned short absIeta = 1; absIeta <= 28; absIeta++)
     {
       EcalSubdetector subdet = ( absIeta <= 17 ) ? EcalBarrel : EcalEndcap ;
       // 8 bits of input energy
       for (unsigned short input = 0; input <= 0xFF; input++)
	 {
	   // negative eta
	   output = ecalTPGScale->
	     getTPGInGeV( (uint) input, EcalTrigTowerDetId(-1, subdet,
							   absIeta, 2));
	   scalesFile << setprecision (8) << output;
	   if (absIeta == 28 && input == 0xFF)
	     {
	       scalesFile << "}";
	     }
	   else
	     {
	       scalesFile << ", ";
	     }
	 }
       scalesFile << endl;
     }

   // Write the hcal scales

     scalesFile << endl << "\tvdouble L1HcalEtThresholds = {" << endl;

   // loop over ietas
   for (unsigned short absIeta = 1; absIeta <= 32; absIeta++)
     {
       for (unsigned short input = 0; input <= 0xFF; input++)
	 {
	   output = caloTPGTranscoder->hcaletValue(absIeta, input); 
	   scalesFile << setprecision (8) << output ;
	   if (absIeta == 32 && input == 0xFF)
	     {
	       scalesFile << "}";
	     }
	   else
	     {
	       scalesFile << ", ";
	     }
	 }
       scalesFile << endl;
     }
   scalesFile << "}" << endl;

   scalesFile.close();
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CaloInputScalesGenerator::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CaloInputScalesGenerator::endJob() {
}

