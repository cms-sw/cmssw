//
//  WaitingTaskList_test.cpp
//  DispatchProcessingDemo
//
//  Created by Chris Jones on 9/27/11.
//

#include <iostream>

#include <cppunit/extensions/HelperMacros.h>
#include <unistd.h>
#include <memory>
#include <atomic>
#include <thread>
#include "tbb/task.h"
#include "boost/shared_ptr.hpp"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)
#define CXX_THREAD_AVAILABLE
#endif

class WaitingTaskList_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(WaitingTaskList_test);
  CPPUNIT_TEST(addThenDone);
  CPPUNIT_TEST(doneThenAdd);
  CPPUNIT_TEST(stressTest);
  CPPUNIT_TEST_SUITE_END();
  
public:
  void addThenDone();
  void doneThenAdd();
  void stressTest();
  void setUp(){}
  void tearDown(){}
};

namespace  {
   class TestCalledTask : public tbb::task {
   public:
      TestCalledTask(std::atomic<bool>& iCalled): m_called(iCalled) {}

      tbb::task* execute() {
         m_called = true;
         return nullptr;
      }
      
   private:
      std::atomic<bool>& m_called;
   };
   
   class TestValueSetTask : public tbb::task {
   public:
      TestValueSetTask(std::atomic<bool>& iValue): m_value(iValue) {}
         tbb::task* execute() {
            CPPUNIT_ASSERT(m_value);
            return nullptr;
         }

      private:
         std::atomic<bool>& m_value;
   };
   
}

void WaitingTaskList_test::addThenDone()
{
   std::atomic<bool> called{false};
   
   edm::WaitingTaskList waitList;
   {
      boost::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                            [](tbb::task* iTask){tbb::task::destroy(*iTask);} };
      waitTask->set_ref_count(2);
      //NOTE: allocate_child does NOT increment the ref_count of waitTask!
      tbb::task* t = new (waitTask->allocate_child()) TestCalledTask{called};
   
      waitList.add(t);

      usleep(10);
      __sync_synchronize();
      CPPUNIT_ASSERT(false==called);
   
      waitList.doneWaiting();
      waitTask->wait_for_all();
      __sync_synchronize();
      CPPUNIT_ASSERT(true==called);
   }
   
   waitList.reset();
   called = false;
   
   {
      boost::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                            [](tbb::task* iTask){tbb::task::destroy(*iTask);} };
      waitTask->set_ref_count(2);
   
      tbb::task* t = new (waitTask->allocate_child()) TestCalledTask{called};
   
      waitList.add(t);

      usleep(10);
      CPPUNIT_ASSERT(false==called);
   
      waitList.doneWaiting();
      waitTask->wait_for_all();
      CPPUNIT_ASSERT(true==called);
   }
}

void WaitingTaskList_test::doneThenAdd()
{
   std::atomic<bool> called{false};
   edm::WaitingTaskList waitList;
   {
      boost::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                            [](tbb::task* iTask){tbb::task::destroy(*iTask);} };
      waitTask->set_ref_count(2);
   
      tbb::task* t = new (waitTask->allocate_child()) TestCalledTask{called};

      waitList.doneWaiting();
   
      waitList.add(t);
      waitTask->wait_for_all();
      CPPUNIT_ASSERT(true==called);
   }
}

namespace {
#if defined(CXX_THREAD_AVAILABLE)
   void join_thread(std::thread* iThread){ 
      if(iThread->joinable()){iThread->join();}
   }
#endif
}

void WaitingTaskList_test::stressTest()
{
#if defined(CXX_THREAD_AVAILABLE)
   std::atomic<bool> called{false};
   edm::WaitingTaskList waitList;
   
   unsigned int index = 1000;
   const unsigned int nTasks = 10000;
   while(0 != --index) {
      called = false;
      boost::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                            [](tbb::task* iTask){tbb::task::destroy(*iTask);} };
      waitTask->set_ref_count(3);
      tbb::task* pWaitTask=waitTask.get();
      
      {
         std::thread makeTasksThread([&waitList,pWaitTask,&called]{
            for(unsigned int i = 0; i<nTasks;++i) {
               auto t = new (tbb::task::allocate_additional_child_of(*pWaitTask)) TestCalledTask{called};
               waitList.add(t);
            }
         
            pWaitTask->decrement_ref_count();
            });
         boost::shared_ptr<std::thread>(&makeTasksThread,join_thread);
         
         std::thread doneWaitThread([&waitList,&called,pWaitTask]{
            called=true;
            waitList.doneWaiting();
            pWaitTask->decrement_ref_count();
            });
         boost::shared_ptr<std::thread>(&doneWaitThread,join_thread);
      }
      waitTask->wait_for_all();
   }
#endif
}


CPPUNIT_TEST_SUITE_REGISTRATION( WaitingTaskList_test );
