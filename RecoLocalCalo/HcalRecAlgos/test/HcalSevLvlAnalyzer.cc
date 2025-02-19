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
// $Id: HcalSevLvlAnalyzer.cc,v 1.4 2009/12/14 22:23:57 wmtan Exp $
//
//


// system include files
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

   for (unsigned i=0; i<4; i++) //loop over rechitflag
     for (unsigned k=0; k<5; k++) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHB, sampleHBRHFlag[i], sampleHBChSt[k]);
	 
	 std::cout << "Status for " << myIdHB 
		   << " with RHFlag " << sampleHBRHFlag[i] << " (" << std::hex << sampleHBRHFlag[i] 
		   << ") and ChStFlag " << std::dec << sampleHBChSt[k] 
		   << " (" << std::hex << sampleHBChSt[k] << std::dec << ") is: " << theLevel;
	 std::cout << std::endl;

	 bool dropchannel = myProd->dropChannel(sampleHBChSt[k]);
	 bool recovered = myProd->recoveredRecHit(myIdHB, sampleHBRHFlag[i]);
	 
	 std::cout << "DropChannel status for " << myIdHB 
		   << " with RHFlag " << sampleHBRHFlag[i] << " (" << std::hex << sampleHBRHFlag[i] 
		   << ") and ChStFlag " << std::dec << sampleHBChSt[k] 
		   << " (" << std::hex << sampleHBChSt[k] << std::dec << ") is: " << dropchannel
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

   for (unsigned i=0; i<4; i++) //loop over rechitflag
     for (unsigned k=0; k<5; k++) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHE, sampleHERHFlag[i], sampleHEChSt[k]);
	 
	 std::cout << "Status for " << myIdHE 
		   << " with RHFlag " << sampleHERHFlag[i] << " (" << std::hex << sampleHERHFlag[i] 
		   << ") and ChStFlag " << std::dec << sampleHEChSt[k] 
		   << " (" << std::hex << sampleHEChSt[k] << std::dec << ") is: " << theLevel;
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

   for (unsigned i=0; i<4; i++) //loop over rechitflag
     for (unsigned k=0; k<5; k++) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHF, sampleHFRHFlag[i], sampleHFChSt[k]);
	 
	 std::cout << "Status for " << myIdHF 
		   << " with RHFlag " << sampleHFRHFlag[i] << " (" << std::hex << sampleHFRHFlag[i] 
		   << ") and ChStFlag " << std::dec << sampleHFChSt[k] 
		   << " (" << std::hex << sampleHFChSt[k] << std::dec << ") is: " << theLevel;
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

   for (unsigned i=0; i<3; i++) //loop over rechitflag
     for (unsigned k=0; k<5; k++) //loop over channelstatus
       {
	 int theLevel = myProd->getSeverityLevel(myIdHO, sampleHORHFlag[i], sampleHOChSt[k]);
	 
	 std::cout << "Status for " << myIdHO 
		   << " with RHFlag " << sampleHORHFlag[i] << " (" << std::hex << sampleHORHFlag[i] 
		   << ") and ChStFlag " << std::dec << sampleHOChSt[k] 
		   << " (" << std::hex << sampleHOChSt[k] << std::dec << ") is: " << theLevel;
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
