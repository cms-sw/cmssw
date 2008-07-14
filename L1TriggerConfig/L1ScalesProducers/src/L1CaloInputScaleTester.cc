#include "L1TriggerConfig/L1ScalesProducers/interface/L1CaloInputScaleTester.h"

// system include files
#include <memory>
#include <iostream>
using std::cout;
using std::endl;

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
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
L1CaloInputScaleTester::L1CaloInputScaleTester(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


L1CaloInputScaleTester::~L1CaloInputScaleTester()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1CaloInputScaleTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

   ESHandle<L1CaloEcalScale> caloEcalScale;
   ESHandle<L1CaloHcalScale> caloHcalScale;
   ESHandle<CaloTPGTranscoder> caloTPGTranscoder;
   iSetup.get<L1CaloEcalScaleRcd>().get(caloEcalScale);
   iSetup.get<L1CaloHcalScaleRcd>().get(caloHcalScale);
   iSetup.get<CaloTPGRecord>().get(caloTPGTranscoder);
   
   EcalTPGScale* ecalTPGScale = new EcalTPGScale();
   ecalTPGScale->setEventSetup(iSetup);

   bool ecalIsConsistent = true;
   bool hcalIsConsistent = true;

   float ecal1; 
   float ecal2;
   float hcal1;
   float hcal2;

   // compare the ecal scales

   // 8 bits of input energy
   for (unsigned short input = 0; input < 0xFF; input++)
     {
       // loop over ietas, barrel first
       for (unsigned short absIeta = 1; absIeta <= 17; absIeta++)
	 {

	   // positive eta
	   ecal1 = ecalTPGScale->getTPGInGeV( (uint) input, 
					      EcalTrigTowerDetId(1, EcalBarrel,
								 absIeta, 1));
	   ecal2 = caloEcalScale->et(input, absIeta, 1);
	   if ( !(ecal1 == ecal2) )
	     {
	       ecalIsConsistent = false;
	     }

	   // negative eta
	   ecal1 = ecalTPGScale->
	     getTPGInGeV( (uint) input, EcalTrigTowerDetId(-1, EcalBarrel,
							   absIeta, 1));
	   ecal2 = caloEcalScale->et(input, absIeta, -1);
	   if ( !(ecal1 == ecal2) )
	     {
	       ecalIsConsistent = false;
	     }
	 }
       // now loop over endcap ietas
       for (unsigned short absIeta = 18; absIeta < 29; absIeta++)
	 {

	   // positive eta
	   ecal1 = ecalTPGScale->
	     getTPGInGeV( (uint) input, EcalTrigTowerDetId(1, EcalEndcap,
							   absIeta, 1));
	   ecal2 = caloEcalScale->et(input, absIeta, 1);
	   if ( !(ecal1 == ecal2) )
	     {
	       ecalIsConsistent = false;
	     }
	   
	   // negative eta
	   ecal1 = ecalTPGScale->
	     getTPGInGeV( (uint) input, EcalTrigTowerDetId(-1, EcalEndcap,
							   absIeta, 1));
	   ecal2 = caloEcalScale->et(input, absIeta, -1);
	   if ( !(ecal1 == ecal2) )
	     {
	       ecalIsConsistent = false;
	     }
	 }
     }

   if (!ecalIsConsistent)
     {
       // do something
       //cout << "WARNING: ECAL scales not consistent!" << endl;
       LogWarning("InconsistentData") << "ECAL scales not consistent!";
     }
   else
     {
       // do something else
       //cout << "ECAL scales okay" << endl;
     }

   // compare the hcal scales

   for (unsigned short input = 0; input < 0xFF; input++)
     {
       // loop over ietas
       for (unsigned short absIeta = 1; absIeta < 33; absIeta++)
	 {
	   hcal1 = caloTPGTranscoder->hcaletValue(absIeta, input); // no eta-
	   hcal2 = caloHcalScale->et(input, absIeta, 1); // sign in transcoder
	   if ( !(hcal1 == hcal2) )
	     {
	       hcalIsConsistent = false;
	     }
	 }
     }
   if (!hcalIsConsistent)
     {
       // do something
       //cout << "WARNING: HCAL scales not consistent!" << endl;
       LogWarning("InconsistentData") << "HCAL scales not consistent!";
     }
   else
     {
       // do something else
       //cout << "HCAL scales okay" << endl;
     }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CaloInputScaleTester::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CaloInputScaleTester::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CaloInputScaleTester);
