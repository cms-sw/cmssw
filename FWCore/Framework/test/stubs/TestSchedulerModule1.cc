/**
   \file
   Test Modules for ScheduleBuilder

   \author Stefano ARGIRO
   \date 19 May 2005
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include <memory>
#include <string>

using namespace edm;


class TestSchedulerModule1 : public EDProducer
{
 public:
  explicit TestSchedulerModule1(ParameterSet const& p):pset_(p){
    produces<edmtest::StringProduct>();
  }

  void produce(Event& e, EventSetup const&);

private:
  ParameterSet pset_;
};


void TestSchedulerModule1::produce(Event& e, EventSetup const&)
{
  std::string myname = pset_.getParameter<std::string>("module_name");
  std::unique_ptr<edmtest::StringProduct> product(new edmtest::StringProduct(myname));
  e.put(std::move(product));
}

DEFINE_FWK_MODULE(TestSchedulerModule1);


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
