/**
   \file
   Test Module for testProductRegistry

   \author Stefano ARGIRO
   \version $Id: TestPRegisterModule2.cc,v 1.3 2006/02/08 00:44:26 wmtan Exp $
   \date 19 May 2005
*/

static const char CVSId[] = "$Id: TestPRegisterModule2.cc,v 1.3 2006/02/08 00:44:26 wmtan Exp $";


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/test/stubs/TestPRegisterModule2.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include <cppunit/extensions/HelperMacros.h>
#include <memory>
#include <string>

using namespace edm;

TestPRegisterModule2::TestPRegisterModule2(edm::ParameterSet const& p){
   produces<edmtest::DoubleProduct>();
}

  void TestPRegisterModule2::produce(Event& e, EventSetup const&)
  {
     std::vector<edm::Provenance const*> plist;
     e.getAllProvenance(plist);

     std::vector<edm::Provenance const*>::const_iterator pd = plist.begin();
     
     CPPUNIT_ASSERT(0 !=plist.size());
     CPPUNIT_ASSERT(2 ==plist.size());
     CPPUNIT_ASSERT(pd != plist.end());
     edmtest::StringProduct stringprod;
     edm::TypeID stringID(stringprod);
     CPPUNIT_ASSERT(stringID.friendlyClassName() == 
                    (*pd)->product.friendlyClassName_);
     CPPUNIT_ASSERT((*pd)->product.module.moduleLabel_=="m1");
     
     ++pd;
     CPPUNIT_ASSERT(pd != plist.end());
     
     edmtest::DoubleProduct dprod;
     edm::TypeID dID(dprod);
     CPPUNIT_ASSERT(dID.friendlyClassName() == 
                    (*pd)->product.friendlyClassName_);
     CPPUNIT_ASSERT((*pd)->product.module.moduleLabel_=="m2");
     
    Handle<edmtest::StringProduct> stringp;
    e.getByLabel("m2",stringp);
    CPPUNIT_ASSERT(stringp->name_=="m1");

     std::auto_ptr<edmtest::DoubleProduct> product(new edmtest::DoubleProduct);
     e.put(product);
  }
