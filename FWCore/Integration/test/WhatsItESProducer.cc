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
// $Id: WhatsItESProducer.cc,v 1.8 2006/02/13 23:01:31 wmtan Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Integration/test/WhatsIt.h"
#include "FWCore/Integration/test/Doodad.h"
#include "FWCore/Integration/test/GadgetRcd.h"


//
// class decleration
//
namespace edmtest {

class WhatsItESProducer : public edm::ESProducer {
   public:
      WhatsItESProducer(const edm::ParameterSet&);
      ~WhatsItESProducer();

      typedef std::auto_ptr<WhatsIt> ReturnType;

      ReturnType produce(const GadgetRcd &);
   private:
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
WhatsItESProducer::WhatsItESProducer(const edm::ParameterSet& /*iConfig*/)
{
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
   iRecord.get(doodad);
   
   std::auto_ptr<WhatsIt> pWhatsIt(new WhatsIt) ;

   pWhatsIt->a = doodad->a;

   return pWhatsIt ;
}
}

using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(WhatsItESProducer);
