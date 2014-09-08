/**
   \file
   Test Modules for testProductRegistry

   \author Stefano ARGIRO
   \date 19 May 2005
*/


#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/test/stubs/TestPRegisterModule1.h"
#include <memory>
#include <string>

using namespace edm;


TestPRegisterModule1::TestPRegisterModule1(edm::ParameterSet const& p):pset_(p){
   produces<edmtest::StringProduct>();
}

void TestPRegisterModule1::produce(Event& e, EventSetup const&)
{
  
  std::string myname = pset_.getParameter<std::string>("@module_label");
  std::unique_ptr<edmtest::StringProduct> product(new edmtest::StringProduct(myname));
  e.put(std::move(product));
}
