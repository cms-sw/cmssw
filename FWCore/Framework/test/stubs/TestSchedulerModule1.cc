/**
   \file
   Test Modules for ScheduleBuilder

   \author Stefano ARGIRO
   \version $Id: TestSchedulerModule1.cc,v 1.1 2005/05/23 09:40:15 argiro Exp $
   \date 19 May 2005
*/


#include "FWCore/CoreFramework/interface/EDProducer.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

#include "FWCore/CoreFramework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

using namespace edm;

static const char CVSId[] = "$Id: TestSchedulerModule1.cc,v 1.1 2005/05/23 09:40:15 argiro Exp $";

class TestSchedulerModule1 : public EDProducer
{
 public:
  explicit TestSchedulerModule1(ParameterSet const& p);

  void produce(Event& e, EventSetup const&);
};

TestSchedulerModule1::TestSchedulerModule1(ParameterSet const& p)
{
  std::cerr << "TestSchedulerModule1 instance created: " << p.getString("module_label")
            << std::endl;
}

void TestSchedulerModule1::produce(Event& e, EventSetup const&)
{

  std::cout << "TestSchedulerModule1 Producing ..." << std::endl;

}

DEFINE_FWK_MODULE(TestSchedulerModule1)






// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
