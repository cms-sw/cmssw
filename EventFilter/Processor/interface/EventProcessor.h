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
//    Revision 1.8  2006/02/15 00:38:40  meschi
//    reflect most recent changes in FW
//
//    Revision 1.7  2006/01/11 00:19:33  meschi
//    improved run end sequence
//
//    Revision 1.6  2006/01/06 11:31:58  meschi
//    added missing InputSource include
//
//    Revision 1.5  2005/12/21 15:42:51  meschi
//    added module web
//
//    Revision 1.4  2005/12/02 16:34:39  meschi
//    removed EventRegistry reference
//
//    Revision 1.3  2005/11/10 14:32:33  meschi
//    cosmetics
//
//    Revision 1.2  2005/10/19 08:52:36  meschi
//    updated to latest for ProductRegistry
//
//    Revision 1.1  2005/10/10 09:54:45  meschi
//    HLT processor initial implementation
//
//

#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/interface/ScheduleBuilder.h"
//#include "FWCore/Framework/interface/ScheduleExecutor.h"
#include "EventFilter/Processor/src/Schedule.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/Actions.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"



#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/InputSource.h" 
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
//#include "FWCore/Framework/interface/EventRegistry.h"
//#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "boost/shared_ptr.hpp"

#include "toolbox/include/Task.h"

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
      
      EventProcessor(unsigned long/*, boost::shared_ptr<edm::InputService> */);
      ~EventProcessor();
      
      void taskWebPage(xgi::Input *, xgi::Output *, const std::string &);
      void moduleWebPage(xgi::Input *, xgi::Output *, const std::string &);
      void init(std::string &/*edm::ParameterSet &*/);
      void suspend(){paused_=true;}
      void resume(){paused_=false; wakeup();}
      inline int svc(){run();return 0;} //final
      void stopEventLoop(unsigned int);
      void toggleOutput();
      void prescaleInput(unsigned int);
      void prescaleOutput(unsigned int);
      void run();

      void beginRun(); 
      bool endRun();
      bool exited() const {return exited_;}

    private:
      
      unsigned long getVersion() { return 0; }
      unsigned long getPass() { return 0; }
      unsigned long tid_;

      boost::shared_ptr<edm::ParameterSet> params_;
      edm::WorkerRegistry                    wreg_;
      edm::SignallingProductRegistry         preg_;
      PathList                workers_;
      boost::shared_ptr<edm::ActivityRegistry> activityRegistry_;
      edm::ServiceToken serviceToken_;
      boost::shared_ptr<edm::InputSource> input_;
      std::auto_ptr<Schedule> sched_;
      edm::eventsetup::EventSetupProvider esp_;    
      
      bool emittedBeginJob_;
      bool running_;
      bool paused_;
      bool exited_;
      unsigned long eventcount;
      edm::ActionTable act_table_;
      std::vector<edm::ModuleDescription> descs_;
      pthread_mutex_t mutex_; //used to synchronize the worker and control thread
      pthread_cond_t exit_;
      friend class FUEventProcessor;

    };
  
}
#endif

