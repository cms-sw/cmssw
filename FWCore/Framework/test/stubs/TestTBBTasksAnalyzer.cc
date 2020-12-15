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
//
//

// system include files
#include <memory>
#include <atomic>
#include <unistd.h>
#include "tbb/task_group.h"
#include "tbb/task_arena.h"

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

//
// class decleration
//

class TestTBBTasksAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TestTBBTasksAnalyzer(const edm::ParameterSet&);
  ~TestTBBTasksAnalyzer() override;

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  virtual void endJob() override;
  unsigned int startTasks(unsigned int iNTasks, unsigned int iSleepTime) const;
  unsigned int m_nTasksToRun;
  unsigned int m_expectedNumberOfSimultaneousTasks;
  unsigned int m_maxCountedTasks;
  unsigned int m_usecondsToSleep;
  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestTBBTasksAnalyzer::TestTBBTasksAnalyzer(const edm::ParameterSet& iConfig)
    : m_nTasksToRun(iConfig.getUntrackedParameter<unsigned int>("numTasksToRun")),
      m_expectedNumberOfSimultaneousTasks(iConfig.getUntrackedParameter<unsigned int>("nExpectedThreads")),
      m_maxCountedTasks(0),
      m_usecondsToSleep(iConfig.getUntrackedParameter<unsigned int>("usecondsToSleep", 100000)) {
  //now do what ever initialization is needed
}

TestTBBTasksAnalyzer::~TestTBBTasksAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TestTBBTasksAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  unsigned int max = startTasks(m_nTasksToRun, m_usecondsToSleep);

  if (max > m_maxCountedTasks) {
    m_maxCountedTasks = max;
  }
}

unsigned int TestTBBTasksAnalyzer::startTasks(unsigned int iNTasks, unsigned int iSleepTime) const {
  std::atomic<unsigned int> count{0};
  std::atomic<unsigned int> maxCount{0};
  tbb::task_group grp;

  for (unsigned int i = 0; i < iNTasks; ++i) {
    grp.run([&]() {
      unsigned int c = ++count;
      while (true) {
        unsigned int mc = maxCount.load();
        if (c > mc) {
          if (maxCount.compare_exchange_strong(mc, c)) {
            break;
          }
        } else {
          break;
        }
      }
      usleep(m_usecondsToSleep);
      --(count);
    });
  }
  grp.wait();
  return maxCount.load();
}

void TestTBBTasksAnalyzer::endJob() {
  if (((m_expectedNumberOfSimultaneousTasks - 1) > m_maxCountedTasks) ||
      (m_maxCountedTasks > m_expectedNumberOfSimultaneousTasks)) {
    throw cms::Exception("WrongNumberOfTasks")
        << "expected " << m_expectedNumberOfSimultaneousTasks << " but instead saw " << m_maxCountedTasks << "\n";
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(TestTBBTasksAnalyzer);
