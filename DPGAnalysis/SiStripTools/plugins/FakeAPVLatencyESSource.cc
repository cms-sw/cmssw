// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      FakeAPVLatencyESSource
// 
/**\class FakeAPVLatencyESSource FakeAPVLatencyESSource.h DPGAnalysis/SiStripTools/plugins/FakeAPVLatencyESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Jan 12 11:26:33 CET 2009
// $Id: FakeAPVLatencyESSource.cc,v 1.1 2009/02/25 12:16:06 venturia Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DPGAnalysis/SiStripTools/interface/APVLatency.h"
#include "DPGAnalysis/SiStripTools/interface/APVLatencyRcd.h"


//
// class decleration
//

class FakeAPVLatencyESSource : public edm::ESProducer {
   public:
      FakeAPVLatencyESSource(const edm::ParameterSet&);
      ~FakeAPVLatencyESSource();

      typedef boost::shared_ptr<APVLatency> ReturnType;

      ReturnType produce(const APVLatencyRcd&);
   private:
      // ----------member data ---------------------------

  int _apvlatency;
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
FakeAPVLatencyESSource::FakeAPVLatencyESSource(const edm::ParameterSet& iConfig):
  _apvlatency(iConfig.getUntrackedParameter<int>("APVLatency"))
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


FakeAPVLatencyESSource::~FakeAPVLatencyESSource()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
FakeAPVLatencyESSource::ReturnType
FakeAPVLatencyESSource::produce(const APVLatencyRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<APVLatency> pAPVLatency(new APVLatency) ;

   pAPVLatency->put(_apvlatency);

   return pAPVLatency ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FakeAPVLatencyESSource);
