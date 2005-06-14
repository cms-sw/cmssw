/**
   \file
   Test Module for testScheduler

   \author Stefano ARGIRO
   \version $Id: TestSchedulerModule2.cc,v 1.4 2005/06/14 00:01:01 wmtan Exp $
   \date 19 May 2005
*/

static const char CVSId[] = "$Id: TestSchedulerModule2.cc,v 1.4 2005/06/14 00:01:01 wmtan Exp $";


#include "FWCore/CoreFramework/interface/EDProducer.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

#include "FWCore/CoreFramework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/CoreFramework/src/ToyProducts.h"
#include <memory>
#include <string>

namespace edm{

  class TestSchedulerModule2 : public EDProducer
  {
  public:
    explicit TestSchedulerModule2(ParameterSet const& p):pset_(p){}

    void produce(Event& e, EventSetup const&);

  private:
    ParameterSet pset_;
  };


  void TestSchedulerModule2::produce(Event& e, EventSetup const&)
  {

    std::string myname = getParameter<std::string>(pset_, "module_name");
    std::auto_ptr<edmtest::StringProduct> product(new edmtest::StringProduct(myname));
    e.put(product);
    
  }

  DEFINE_FWK_MODULE(TestSchedulerModule2)
}//namespace  

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
