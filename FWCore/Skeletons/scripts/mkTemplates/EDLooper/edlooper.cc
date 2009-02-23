// -*- C++ -*-
//
// Package:    loopername
// Class:      loopername
// 
/**\class loopername loopername.h skelsubsys/loopername/interface/loopername.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"

#include "FWCore/Framework/interface/ESHandle.h"

@perl if( 1 lt scalar( @::datatypes ) ) {$result="#include \"FWCore/Framework/interface/ESProducts.h\""; } @\perl


//
// class decleration
//

class loopername : public edm::ESProducerLooper {
   public:
      loopername(const edm::ParameterSet&);
      ~loopername();

      typedef @perl if( 1 eq scalar( @::datatypes ) ) { $result="std::auto_ptr<$::datatypes[0]>"; } else { $result="edm::ESProducts<"; $line = 0; foreach $type ( @::datatypes ) { if ($line) { $result = "$result, "; } $result= "$result $type";  $line =1;} $result="$result>"; }  @\perl ReturnType;

      ReturnType produce(const recordname&);

      virtual void beginOfJob(); 
      virtual void startingNewLoop(unsigned int ) ; 
      virtual Status duringLoop(const edm::Event&, const edm::EventSetup&) ; 
      virtual Status endOfLoop(const edm::EventSetup&); 
      virtual void endOfJob();
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
loopername::loopername(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


loopername::~loopername()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
loopername::ReturnType
loopername::produce(const recordname& iRecord)
{
   using namespace edm::es;
@perl $result=""; foreach $type (@::datatypes) {$result ="$result   std::auto_ptr<$type> p$type ;\n";} @\perl

   return @perl if( 1 eq scalar( @::datatypes ) ) { $result="p$::datatypes[0]" } else { $result="products("; $line = 0; foreach $type ( @::datatypes ) { if ($line) { $result = "$result,"; } $result= "$result $type"; $line +=1; } $result="$result)"; }  @\perl ;
}


// ------------ method called once per job just before starting to loop over events  ------------
void 
loopername::beginOfJob(const edm::EventSetup&)
{
}

// ------------ method called at the beginning of a new loop over the event  ------------
// ------------ the argument starts at 0 and increments for each loop        ------------
void 
loopername::startingNewLoop(unsigned int iIteration)
{
}

// ------------ called for each event in the loop.  The present event loop can be stopped by return kStop ------------
loopername::Status 
loopername::duringLoop(const edm::Event&, const edm::EventSetup&)
{
  return kContinue;
}


// ------------ called at the end of each event loop. A new loop will occur if you return kContinue ------------
loopername::Status 
loopername::endOfLoop(const edm::EventSetup&, unsigned int)
{
  return kStop;
}

// ------------ called once each job just before the job ends ------------
void 
loopername::endOfJob()
{
}

//define this as a plug-in
DEFINE_FWK_LOOPER(loopername);
