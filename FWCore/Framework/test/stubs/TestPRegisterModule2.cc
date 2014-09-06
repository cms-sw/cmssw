/**
   \file
   Test Module for testProductRegistry

   \author Stefano ARGIRO
   \date 19 May 2005
*/

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/test/stubs/TestPRegisterModule2.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "cppunit/extensions/HelperMacros.h"
#include <cassert>
#include <memory>
#include <string>

using namespace edm;

TestPRegisterModule2::TestPRegisterModule2(edm::ParameterSet const&){
   produces<edmtest::DoubleProduct>();
   consumes<edmtest::StringProduct>(edm::InputTag{"m2"});
}

  void TestPRegisterModule2::produce(Event& e, EventSetup const&)
  {
     std::vector<edm::Provenance const*> plist;
     e.getAllProvenance(plist);

     std::vector<edm::Provenance const*>::const_iterator pd = plist.begin();
     
     CPPUNIT_ASSERT(0 !=plist.size());
     CPPUNIT_ASSERT(2 ==plist.size());
     CPPUNIT_ASSERT(pd != plist.end());
     if(pd == plist.end()) return; // To silence Coverity
     edmtest::StringProduct stringprod;
     edm::TypeID stringID(stringprod);
     CPPUNIT_ASSERT(stringID.friendlyClassName() == 
                    (*pd)->friendlyClassName());
     CPPUNIT_ASSERT((*pd)->moduleLabel()=="m1");
     CPPUNIT_ASSERT((*pd)->releaseVersion()==getReleaseVersion());

     ++pd;
     CPPUNIT_ASSERT(pd != plist.end());
     if(pd == plist.end()) return; // To silence Coverity
     
     edmtest::DoubleProduct dprod;
     edm::TypeID dID(dprod);
     CPPUNIT_ASSERT(dID.friendlyClassName() == 
                  (*pd)->friendlyClassName());
     CPPUNIT_ASSERT((*pd)->moduleLabel()=="m2");
     
     Handle<edmtest::StringProduct> stringp;
     e.getByLabel("m2",stringp);
     CPPUNIT_ASSERT(stringp->name_=="m1");

     std::unique_ptr<edmtest::DoubleProduct> product(new edmtest::DoubleProduct);
     e.put(std::move(product));
  }
