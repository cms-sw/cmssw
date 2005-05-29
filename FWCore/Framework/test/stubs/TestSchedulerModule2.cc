/**
   \file
   Test Module for testScheduler

   \author Stefano ARGIRO
   \version $Id: TestSchedulerModule2.cc,v 1.1 2005/05/23 09:40:15 argiro Exp $
   \date 19 May 2005
*/

static const char CVSId[] = "$Id: TestSchedulerModule2.cc,v 1.1 2005/05/23 09:40:15 argiro Exp $";


#include "FWCore/CoreFramework/interface/EDProducer.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

#include "FWCore/CoreFramework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

using namespace edm;

class TestSchedulerModule2 : public EDProducer
{
 public:
  explicit TestSchedulerModule2(ParameterSet const& p);

  void produce(Event& e, EventSetup const&);
};

TestSchedulerModule2::TestSchedulerModule2(ParameterSet const& p)
{
  std::cerr << "TestSchedulerModule2 instance created: " << 
    p.getString(std::string("module_label"))<< std::endl;
}

void TestSchedulerModule2::produce(Event& e, EventSetup const&)
{

  std::cout << "TestSchedulerModule2 Producing ..." << std::endl;

}

DEFINE_FWK_MODULE(TestSchedulerModule2)


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
