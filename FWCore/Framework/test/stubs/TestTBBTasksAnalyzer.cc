// -*- C++ -*-
//
// Package:    Framework
// Class:      TestTBBTasksAnalyzer
// 
/**\class TestTBBTasksAnalyzer TestTBBTasksAnalyzer.cc FWCore/Framework/test/stubs/TestTBBTasksAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 03 11:02:00 EST 2013
// $Id: TestTBBTasksAnalyzer.cc,v 1.7 2007/08/07 22:34:20 wmtan Exp $
//
//


// system include files
#include <memory>
#include <atomic>
#include <unistd.h>
#include "tbb/task.h"

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

//
// class decleration
//

class TestTBBTasksAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestTBBTasksAnalyzer(const edm::ParameterSet&);
      ~TestTBBTasksAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
         virtual void endJob();
         unsigned int m_nTasksToRun;
         unsigned int m_expectedNumberOfSimultaneousTasks;
         unsigned int m_maxCountedTasks;
         unsigned int m_usecondsToSleep;
      // ----------member data ---------------------------
};

namespace {
   class WaitTask : public tbb::task {
   public:
      WaitTask(unsigned int iSleepUSecs, std::atomic<unsigned int>* iCount, std::atomic<unsigned int>* iMaxCount): m_usecondsToSleep(iSleepUSecs),m_count(iCount),m_maxCount(iMaxCount) {}
      tbb::task* execute() {
         unsigned int c = ++(*m_count);
         __sync_synchronize();
         while(true) {
            unsigned int mc = *m_maxCount;
            if(c > mc) {
               if(m_maxCount->compare_exchange_strong(mc,c)) {
                  break;
               }
            }else {
               break;
            }
         }
         usleep(m_usecondsToSleep);
         --(*m_count);
         return 0;
      }
   private:
      unsigned int m_usecondsToSleep;
      std::atomic<unsigned int>* m_count;
      std::atomic<unsigned int>* m_maxCount;
   };

   unsigned int startTasks(unsigned int iNTasks, unsigned int iSleepTime) {
      std::atomic<unsigned int> count{0};
      std::atomic<unsigned int> maxCount{0};
      tbb::task* sync = new(tbb::task::allocate_root()) tbb::empty_task;
      sync->set_ref_count(iNTasks+1);
      for(unsigned int i=0; i<iNTasks;++i){
         tbb::task* t = new(sync->allocate_child()) WaitTask(iSleepTime,&count,&maxCount);
         sync->spawn(*t);
      }
      sync->wait_for_all();
      sync->destroy(*sync);
      return maxCount.load();
   }

}
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestTBBTasksAnalyzer::TestTBBTasksAnalyzer(const edm::ParameterSet& iConfig) :
m_nTasksToRun(iConfig.getUntrackedParameter<unsigned int>("numTasksToRun")),
m_expectedNumberOfSimultaneousTasks(iConfig.getUntrackedParameter<unsigned int>("nExpectedThreads")),
m_maxCountedTasks(0),
m_usecondsToSleep(iConfig.getUntrackedParameter<unsigned int>("usecondsToSleep",100000) )
{
   //now do what ever initialization is needed

}


TestTBBTasksAnalyzer::~TestTBBTasksAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TestTBBTasksAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iSetup)
{

   unsigned int max = startTasks(m_nTasksToRun,m_usecondsToSleep);
   
   if(max > m_maxCountedTasks) {
      m_maxCountedTasks = max;
   }
   
}

void 
TestTBBTasksAnalyzer::endJob()
{
  if (m_maxCountedTasks != m_expectedNumberOfSimultaneousTasks) {
    throw cms::Exception("WrongNumberOfTasks")<<"expected "<<m_expectedNumberOfSimultaneousTasks<<" but instead saw "<<m_maxCountedTasks
					       <<"\n";
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(TestTBBTasksAnalyzer);
