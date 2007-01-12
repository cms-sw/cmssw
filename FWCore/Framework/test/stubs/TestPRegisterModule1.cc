/**
   \file
   Test Modules for testProductRegistry

   \author Stefano ARGIRO
   \version $Id: TestPRegisterModule1.cc,v 1.5 2006/03/06 01:13:36 chrjones Exp $
   \date 19 May 2005
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/test/stubs/TestPRegisterModule1.h"
#include <memory>
#include <string>

using namespace edm;

static const char CVSId[] = "$Id: TestPRegisterModule1.cc,v 1.5 2006/03/06 01:13:36 chrjones Exp $";

TestPRegisterModule1::TestPRegisterModule1(edm::ParameterSet const& p):pset_(p){
   produces<edmtest::StringProduct>();
}

void TestPRegisterModule1::produce(Event& e, EventSetup const&)
{
  
  std::string myname = pset_.getParameter<std::string>("@module_label");
  std::auto_ptr<edmtest::StringProduct> product(new edmtest::StringProduct(myname)); 
  e.put(product);
}
