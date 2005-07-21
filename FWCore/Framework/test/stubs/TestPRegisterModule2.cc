/**
   \file
   Test Module for testProductRegistry

   \author Stefano ARGIRO
   \version $Id: TestSchedulerModule2.cc,v 1.8 2005/07/14 22:50:53 wmtan Exp $
   \date 19 May 2005
*/

static const char CVSId[] = "$Id: TestPRegisterModule2.cc,v 1.8 2005/07/14 22:50:53 wmtan Exp $";


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/src/ToyProducts.h"
#include <cppunit/extensions/HelperMacros.h>
#include <memory>
#include <string>

using namespace edm;

  class TestPRegisterModule2 : public EDProducer
  {
  public:
    explicit TestPRegisterModule2(ParameterSet const& p):pset_(p){
      produces<edmtest::DoubleProduct>();
    }

    void produce(Event& e, EventSetup const&);

  private:
    ParameterSet pset_;
  };


  void TestPRegisterModule2::produce(Event& e, EventSetup const&)
  {

    Handle<edmtest::StringProduct> stringp;
    e.getByLabel("m2",stringp);
    CPPUNIT_ASSERT(stringp->name_=="m1");

     std::auto_ptr<edmtest::DoubleProduct> product(new edmtest::DoubleProduct);
     e.put(product);
    
  }
 

DEFINE_FWK_MODULE(TestPRegisterModule2)
