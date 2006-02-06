// -*- C++ -*-
//
// Package:    MCJet
// Class:      MCJet
// 
/**\class MCJet MCJet.cc JetMETCorrections/MCJet/src/MCJet.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Olga Kodolova
//         Created:  Wed Feb  1 17:04:23 CET 2006
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "JetMETCorrections/MCJet/interface/JetCalibratorMCJet.h"
#include "CLHEP/Vector/LorentzVector.h"

namespace cms 
{

//
// class decleration
//

class MCJet : public edm::EDProducer {
   public:
      explicit MCJet(const edm::ParameterSet&);
      ~MCJet();
      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
   JetCalibratorMCJet alg_; 
   std::string inputLabel;  
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
MCJet::MCJet(const edm::ParameterSet& iConfig): 
                                           alg_()
{
   //register your products
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
#endif

   //now do what ever other initialization is needed
    produces<CaloJetCollection>();
    //inputLabel = iConfig.getParameter<std::string>("inputLabel");
    
    std::string newtag = iConfig.getParameter<std::string>("tagName");
    //std::cout<<" Read the tag from txt file "<<newtag<<endl;
    alg_.setParameters(newtag);
}


MCJet::~MCJet()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MCJet::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<CaloJetCollection> pIn;                                                   //Define Inputs
   //iEvent.getByLabel(inputLabel, pIn);                                            //Get Inputs
   iEvent.getByType(pIn);                                                           //Get Inputs
   std::auto_ptr<std::vector<HepLorentzVector> > pOut( new std::vector<HepLorentzVector>() );         //Create empty output
   alg_.run( pIn.product(), *pOut );                                                //Invoke the algorithm
   iEvent.put( pOut );                                                              //Put output into Event
   
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCJet)
    }//end namespace cms
