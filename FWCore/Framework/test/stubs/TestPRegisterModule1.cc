/**
   \file
   Test Modules for testProductRegistry

   \author Stefano ARGIRO
   \version $Id: TestPRegisterModule1.cc,v 1.2 2005/09/28 17:32:55 chrjones Exp $
   \date 19 May 2005
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ToyProducts.h"
#include "FWCore/Framework/test/stubs/TestPRegisterModule1.h"
#include <memory>
#include <string>

using namespace edm;

static const char CVSId[] = "$Id: TestPRegisterModule1.cc,v 1.2 2005/09/28 17:32:55 chrjones Exp $";

TestPRegisterModule1::TestPRegisterModule1(edm::ParameterSet const& p):pset_(p){
   produces<edmtest::StringProduct>();
}

void TestPRegisterModule1::produce(Event& e, EventSetup const&)
{
  
  std::string myname = pset_.getParameter<std::string>("module_name");
  std::auto_ptr<edmtest::StringProduct> product(new edmtest::StringProduct(myname)); 
  e.put(product);
}
