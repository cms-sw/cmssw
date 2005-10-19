#ifndef FPEventProcessor_H
#define FPEventProcessor_H
//
///
//
//
/// Author: Emilio Meschi, PH/CMD
/// 
//
//  MODIFICATION:
//    $Log: EventProcessor.h,v $
//    Revision 1.1  2005/10/10 09:54:45  meschi
//    HLT processor initial implementation
//
//

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/interface/ScheduleBuilder.h"
#include "FWCore/Framework/interface/ScheduleExecutor.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/Actions.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"



#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventRegistry.h"
//#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "boost/shared_ptr.hpp"

#include "toolbox/include/Task.h"
#include "toolbox/include/BSem.h"

#include "xgi/include/xgi/Input.h"
#include "xgi/include/xgi/Output.h"


#include <string>
#include <vector>
#include <list>

namespace evf
{
  typedef edm::Worker* WorkerPtr;
  typedef std::list<WorkerPtr> WorkerList;
  typedef std::list<WorkerList> PathList;

  class EventProcessor : public Task
    {
      
    public:
      
      EventProcessor(int/*, boost::shared_ptr<edm::InputService> */);
      ~EventProcessor();
      
      void taskWebPage(xgi::Input *, xgi::Output *);
      void init(std::string &/*edm::ParameterSet &*/);
      void suspend(){paused_=true;}
      void resume(){paused_=false; wakeup();}
      inline int svc(){run();return 0;} //final
      void stopEventLoop(){running_ = false;}
      void run();

      void beginRun(); 
      bool endRun();
      
    private:
      
      unsigned long getVersion() { return 0; }
      unsigned long getPass() { return 0; }
      int tid_;

      boost::shared_ptr<edm::ParameterSet> params_;
      edm::WorkerRegistry                    wreg_;
      edm::SignallingProductRegistry         preg_;
      PathList                workers_;
      edm::ActivityRegistry activityRegistry_;
      edm::ServiceToken serviceToken_;
      boost::shared_ptr<edm::InputSource> input_;
      std::auto_ptr<edm::ScheduleExecutor> runner_;
      edm::eventsetup::EventSetupProvider esp_;    
      
      bool emittedBeginJob_;
      bool running_;
      bool paused_;
      unsigned long eventcount;
      edm::ActionTable act_table_;
      std::vector<edm::ModuleDescription> descs_;
      friend class FUEventProcessor;
      friend int main(int argc, char* argv[]);
    };
  
}
#endif

