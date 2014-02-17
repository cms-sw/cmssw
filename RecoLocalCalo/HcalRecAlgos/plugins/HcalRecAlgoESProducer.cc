// -*- C++ -*-
//
// Package:    HcalRecAlgoESProducer
// Class:      HcalRecAlgoESProducer
// 
/**\class HcalRecAlgoESProducer HcalRecAlgoESProducer.h TestSubsystem/HcalRecAlgoESProducer/src/HcalRecAlgoESProducer.cc

 Description: Producer for HcalSeverityLevelComputer, that delivers the severity level for HCAL cells

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Radek Ofierzynski
//         Created:  Mon Feb  9 10:59:46 CET 2009
// $Id: HcalRecAlgoESProducer.cc,v 1.1 2009/02/09 16:51:44 rofierzy Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"



//
// class decleration
//

class HcalRecAlgoESProducer : public edm::ESProducer {
   public:
      HcalRecAlgoESProducer(const edm::ParameterSet&);

      ~HcalRecAlgoESProducer();

      typedef boost::shared_ptr<HcalSeverityLevelComputer> ReturnType;

      ReturnType produce(const HcalSeverityLevelComputerRcd&);
   private:
      // ----------member data ---------------------------
  ReturnType myComputer;
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
HcalRecAlgoESProducer::HcalRecAlgoESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
   myComputer = ReturnType(new HcalSeverityLevelComputer(iConfig));
}


HcalRecAlgoESProducer::~HcalRecAlgoESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalRecAlgoESProducer::ReturnType
HcalRecAlgoESProducer::produce(const HcalSeverityLevelComputerRcd& iRecord)
{
   using namespace edm::es;

   return myComputer ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalRecAlgoESProducer);
