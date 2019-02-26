// -*- C++ -*-
//
// Package:    __subsys__/__pkgname__
// Class:      __class__
// 
/**\class __class__

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  __author__
//         Created:  __date__
//
//

// PLEASE DELETE COMMENTS THAT THE SKELETON METHOD (mkesprod)
// GENERATES THAT ARE NOT USEFUL FOR LONG TERM CODE MAINTENANCE
// AND UNDERSTANDING. (For example, please delete this comment)

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#python_begin
    if  len(__datatypes__) > 1:
        print('#include "FWCore/Framework/interface/ESProducts.h"')
#python_end

// Need to add #include statements for definitions of
// the data type and record type here

//
// class declaration
//

class __class__ : public edm::ESProducer {
   public:
      __class__(const edm::ParameterSet&);
      ~__class__();

#python_begin
    if  len(__datatypes__) > 1:
        datatypes = []
        for dtype in __datatypes__:
            datatypes.append("std::unique_ptr<%s>" % dtype)
        print("      using ReturnType = edm::ESProducts<%s>;" % ','.join(datatypes))
    elif len(__datatypes__) == 1:
        print("      using ReturnType = std::unique_ptr<%s>;" % __datatypes__[0])
#python_end

      ReturnType produce(const __record__&);
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
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}

//
// member functions
//

// ------------ method called to produce the data  ------------
__class__::ReturnType
__class__::produce(const __record__& iRecord)
{
   // You can add arguments to the make_unique function call
   // and they will be forwarded to the constructor of the
   // data object. Also you can call functions that modify
   // the data object after creating it. Often, before this
   // you will retrieve data from the EventSetup through the
   // record.
#python_begin
    if  len(__datatypes__) > 1:
        out1 = []
        out2 = []
        i = 1
        for dtype in __datatypes__:
            out1.append("   auto p%s = std::make_unique<%s>();" % (i, dtype))
            out2.append("std::move(p%s)" % i)
            i = i + 1
        output  = '\n'.join(out1)
        output += "\n   return edm::es::products(%s);" % ','.join(out2)
        print(output)
    elif len(__datatypes__) == 1:
        print("   auto product = std::make_unique<%s>();" % __datatypes__[0])
        print("   return product;")
#python_end
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(__class__);
