// -*- C++ -*-
//
// Package:    HcalSevLvlAnalyzer
// Class:      HcalSevLvlAnalyzer
// 
/**\class HcalSevLvlAnalyzer HcalSevLvlAnalyzer.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Radek Ofierzynski
//         Created:  Wed Jan 21 13:46:27 CET 2009
//
//


// system include files
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
//
// class decleration
//

class HcalSevLvlAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HcalSevLvlAnalyzer(const edm::ParameterSet&);
      ~HcalSevLvlAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalSevLvlAnalyzer::HcalSevLvlAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

  // initialize the severity level code

}


HcalSevLvlAnalyzer::~HcalSevLvlAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HcalSevLvlAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

   // here's how to access the severity level computer:
   ESHandle<HcalSeverityLevelComputer> mycomputer;
   iSetup.get<HcalSeverityLevelComputerRcd>().get(mycomputer);
   const HcalSeverityLevelComputer* myProd = mycomputer.product();



   // create some example cases:
   std::cout << std::showbase;
   //HB
   HcalGenericDetId myIdHB(1107313548);
   uint32_t sampleHBChSt[5] = {0, //nothing
			     4, //nothing significant
			     64, //hot
			     1, //off
			     32};  //dead
   uint32_t sampleHBRHFlag[4] = {0,  //nothing
			       32,  //nothing significant
			       2,  //Pulseshape
			       1};  //Multiplicity

   for (unsigned int i : sampleHBRHFlag) //loop over rechitflag
     for (unsigned int k : sampleHBChSt) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHB, i, k);
	 
	 std::cout << "Status for " << myIdHB 
		   << " with RHFlag " << i << " (" << std::hex << i 
		   << ") and ChStFlag " << std::dec << k 
		   << " (" << std::hex << k << std::dec << ") is: " << theLevel;
	 std::cout << std::endl;

	 bool dropchannel = myProd->dropChannel(k);
	 bool recovered = myProd->recoveredRecHit(myIdHB, i);
	 
	 std::cout << "DropChannel status for " << myIdHB 
		   << " with RHFlag " << i << " (" << std::hex << i 
		   << ") and ChStFlag " << std::dec << k 
		   << " (" << std::hex << k << std::dec << ") is: " << dropchannel
		   << ", recovered status is: " << recovered << std::endl;
	 std::cout << std::endl;

       }
   std::cout << std::endl;

//   sampleChSt = 64; //hot
//   sampleRHFlag = 2; //HBHEPulseShape
//   theLevel = myProd->getSeverityLevel(myId, sampleRHFlag, sampleChSt);
//   std::cout << "Status for " << myId << " with RHFlag " <<  std::hex << sampleRHFlag 
//	     << " and ChStFlag " << sampleChSt << std::dec << " is: " << theLevel << std::endl;
//   std::cout << std::endl;

   //HE
   HcalGenericDetId myIdHE(1140869515);
   uint32_t sampleHEChSt[5] = {0, //nothing
			     4, //nothing significant
			     64, //hot
			     1, //off
			     32};  //dead
   uint32_t sampleHERHFlag[4] = {0,  //nothing
			       32,  //nothing significant
			       2,  //Pulseshape
			       1};  //Multiplicity

   for (unsigned int i : sampleHERHFlag) //loop over rechitflag
     for (unsigned int k : sampleHEChSt) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHE, i, k);
	 
	 std::cout << "Status for " << myIdHE 
		   << " with RHFlag " << i << " (" << std::hex << i 
		   << ") and ChStFlag " << std::dec << k 
		   << " (" << std::hex << k << std::dec << ") is: " << theLevel;
	 std::cout << std::endl; 
       }
   std::cout << std::endl;

   //HF
   HcalGenericDetId myIdHF(1207979653);
   uint32_t sampleHFChSt[5] = {0, //nothing
			     4, //nothing significant
			     64, //hot
			     1, //off
			     32};  //dead
   uint32_t sampleHFRHFlag[4] = {0,  //nothing
			       32,  //nothing significant
			       1,  //HFDigiTime
			       2};  //HFLongShort

   for (unsigned int i : sampleHFRHFlag) //loop over rechitflag
     for (unsigned int k : sampleHFChSt) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHF, i, k);
	 
	 std::cout << "Status for " << myIdHF 
		   << " with RHFlag " << i << " (" << std::hex << i 
		   << ") and ChStFlag " << std::dec << k 
		   << " (" << std::hex << k << std::dec << ") is: " << theLevel;
	 std::cout << std::endl; 
       }
   std::cout << std::endl;

   //HO
   HcalGenericDetId myIdHO(1174471682);
   uint32_t sampleHOChSt[5] = {0, //nothing
			     4, //nothing significant
			     64, //hot
			     1, //off
			     32};  //dead
   uint32_t sampleHORHFlag[3] = {0,  //nothing
			       32,  //nothing significant
			       1};  //HOBit

   for (unsigned int i : sampleHORHFlag) //loop over rechitflag
     for (unsigned int k : sampleHOChSt) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHO, i, k);
	 
	 std::cout << "Status for " << myIdHO 
		   << " with RHFlag " << i << " (" << std::hex << i 
		   << ") and ChStFlag " << std::dec << k 
		   << " (" << std::hex << k << std::dec << ") is: " << theLevel;
	 std::cout << std::endl; 
       }
   std::cout << std::endl;

   //ZDC

   //Calib


   std::cout << std::noshowbase;

}


// ------------ method called once each job just before starting event loop  ------------
void 
HcalSevLvlAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalSevLvlAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalSevLvlAnalyzer);
