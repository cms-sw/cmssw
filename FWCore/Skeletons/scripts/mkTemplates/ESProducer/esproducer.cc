// -*- C++ -*-
//
// Package:    prodname
// Class:      prodname
// 
/**\class prodname prodname.h skelsubsys/prodname/src/prodname.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
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
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

@perl if( 1 lt scalar( @::datatypes ) ) {$result="#include \"FWCore/Framework/interface/ESProducts.h\""; } @\perl


//
// class declaration
//

class prodname : public edm::ESProducer {
   public:
      prodname(const edm::ParameterSet&);
      ~prodname();

      typedef @perl if( 1 eq scalar( @::datatypes ) ) { $result="boost::shared_ptr<$::datatypes[0]>"; } else { $result="edm::ESProducts<"; $line = 0; foreach $type ( @::datatypes ) { if ($line) { $result = "$result, "; } $result= "$result boost::shared_ptr<$type> ";  $line =1;} $result="$result>"; }  @\perl ReturnType;

      ReturnType produce(const recordname&);
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
prodname::prodname(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


prodname::~prodname()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
prodname::ReturnType
prodname::produce(const recordname& iRecord)
{
   using namespace edm::es;
@perl $result=""; foreach $type (@::datatypes) {$result ="$result   boost::shared_ptr<$type> p$type ;\n";} @\perl

   return @perl if( 1 eq scalar( @::datatypes ) ) { $result="p$::datatypes[0]" } else { $result="products("; $line = 0; foreach $type ( @::datatypes ) { if ($line) { $result = "$result,"; } $result= "$result p$type"; $line +=1; } $result="$result)"; }  @\perl ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(prodname);
