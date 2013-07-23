// -*- C++ -*-
//
// Package:    WhatsItESProducer
// Class:      WhatsItESProducer
// 
/**\class WhatsItESProducer WhatsItESProducer.h test/WhatsItESProducer/interface/WhatsItESProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 14:33:04 EDT 2005
// $Id: WhatsItESProducer.cc,v 1.11 2007/08/08 16:44:49 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Integration/test/WhatsIt.h"
#include "FWCore/Integration/test/Doodad.h"
#include "FWCore/Integration/test/GadgetRcd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

//
// class decleration
//
namespace edmtest {

class WhatsItESProducer : public edm::ESProducer {
   public:
      WhatsItESProducer(edm::ParameterSet const& pset);
      ~WhatsItESProducer();

      typedef std::auto_ptr<WhatsIt> ReturnType;

      ReturnType produce(const GadgetRcd &);

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      // ----------member data ---------------------------
      std::string dataLabel_;
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
WhatsItESProducer::WhatsItESProducer(edm::ParameterSet const& pset)
: dataLabel_(pset.exists("doodadLabel")? pset.getParameter<std::string>("doodadLabel"):std::string(""))
{
  if (pset.getUntrackedParameter<bool>("test", true)) {
     throw edm::Exception(edm::errors::Configuration, "Something is wrong with ESProducer validation\n")
       << "Or the test configuration parameter was set true (it should never be true unless you want this exception)\n";
   }

   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


WhatsItESProducer::~WhatsItESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
WhatsItESProducer::ReturnType
WhatsItESProducer::produce(const GadgetRcd& iRecord)
{
   using namespace edmtest;

   edm::ESHandle<Doodad> doodad;
   iRecord.get(dataLabel_,doodad);
   
   std::auto_ptr<WhatsIt> pWhatsIt(new WhatsIt) ;

   pWhatsIt->a = doodad->a;

   return pWhatsIt ;
}

void
WhatsItESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addOptional<std::string>("doodadLabel");
  desc.addUntracked<bool>("test", false)->
    setComment("This parameter exists only to test the parameter set validation for ESSources"); 
  descriptions.add("WhatsItESProducer", desc);
}
}
using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(WhatsItESProducer);
