// -*- C++ -*-
//
// Package:    __subsys__/__pkgname__
// Class:      __class__
// 
/**\class __class__ __class__.cc __subsys__/__pkgname__/plugins/__class__.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  __author__
//         Created:  __date__
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducts.h"


//
// class declaration
//

class __class__ : public edm::ESProducerLooper {
   public:
      __class__(const edm::ParameterSet&);
      ~__class__();

#python_begin
    if  len(__datatypes__) > 1:
        datatypes = []
        for dtype in __datatypes__:
            datatypes.append("boost::auto_ptr<%s>" % dtype)
        print "      typedef edm::ESProducts<%s> ReturnType;" % ','.join(datatypes)
    elif len(__datatypes__) == 1:
        print "      typedef std::shared_ptr<%s> ReturnType;" % __datatypes__[0]
#python_end

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
__class__::__class__(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


__class__::~__class__()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
__class__::ReturnType
__class__::produce(const recordname& iRecord)
{
   using namespace edm::es;
#python_begin
    out1 = []
    out2 = []
    for dtype in __datatypes__:
        out1.append("   std::auto_ptr<%s> p%s;" % (dtype, dtype))
        out2.append("p%s" % dtype)
    output  = '\n'.join(out1)
    output += "\n   return products(%s);\n" % ','.join(out2)
    print output
#python_end
}


// ------------ method called once per job just before starting to loop over events  ------------
void 
__class__::beginOfJob(const edm::EventSetup&)
{
}

// ------------ method called at the beginning of a new loop over the event  ------------
// ------------ the argument starts at 0 and increments for each loop        ------------
void 
__class__::startingNewLoop(unsigned int iIteration)
{
}

// ------------ called for each event in the loop.  The present event loop can be stopped by return kStop ------------
__class__::Status 
__class__::duringLoop(const edm::Event&, const edm::EventSetup&)
{
  return kContinue;
}


// ------------ called at the end of each event loop. A new loop will occur if you return kContinue ------------
__class__::Status 
__class__::endOfLoop(const edm::EventSetup&, unsigned int)
{
  return kStop;
}

// ------------ called once each job just before the job ends ------------
void 
__class__::endOfJob()
{
}

//define this as a plug-in
DEFINE_FWK_LOOPER(__class__);
