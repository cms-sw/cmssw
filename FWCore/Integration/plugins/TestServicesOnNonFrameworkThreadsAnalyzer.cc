#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/edm_MessageLogger.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>
#include <exception>

#include "CLHEP/Random/RandFlat.h"

namespace edmtest {
  class TestServicesOnNonFrameworkThreadsAnalyzer : public edm::stream::EDAnalyzer<> {
  public:
    TestServicesOnNonFrameworkThreadsAnalyzer(edm::ParameterSet const&);
    ~TestServicesOnNonFrameworkThreadsAnalyzer() override;

    void analyze(edm::Event const&, edm::EventSetup const&) final;

  private:
    void runOnOtherThread();
    void shutdownThread();
    std::unique_ptr<std::thread> m_thread;
    std::mutex m_mutex;
    std::condition_variable m_condVar;

    bool m_managerThreadReady = false;
    bool m_continueProcessing = false;
    bool m_eventWorkDone = false;

    //context info
    edm::ModuleCallingContext const* m_moduleCallingContext = nullptr;
    edm::ServiceToken* m_serviceToken = nullptr;
    edm::StreamID m_streamID;
    std::exception_ptr m_except;
  };

  TestServicesOnNonFrameworkThreadsAnalyzer::TestServicesOnNonFrameworkThreadsAnalyzer(edm::ParameterSet const&)
      : m_streamID(edm::StreamID::invalidStreamID()) {
    m_thread = std::make_unique<std::thread>([this]() { this->runOnOtherThread(); });

    m_mutex.lock();
    m_managerThreadReady = true;
    m_continueProcessing = true;
  }

  TestServicesOnNonFrameworkThreadsAnalyzer::~TestServicesOnNonFrameworkThreadsAnalyzer() {
    if (m_thread) {
      shutdownThread();
    }
  }

  void TestServicesOnNonFrameworkThreadsAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
    m_eventWorkDone = false;
    m_moduleCallingContext = iEvent.moduleCallingContext();
    edm::ServiceToken token = edm::ServiceRegistry::instance().presentToken();
    m_serviceToken = &token;
    m_streamID = iEvent.streamID();
    { edm::LogSystem("FrameworkThread") << "new Event"; }
    m_mutex.unlock();
    {
      std::unique_lock<std::mutex> lk(m_mutex);
      m_condVar.notify_one();
      m_condVar.wait(lk, [this] { return this->m_eventWorkDone; });
      lk.release();
    }
    edm::LogSystem("FrameworkThread") << " done";
    m_managerThreadReady = true;
    if (m_except) {
      std::rethrow_exception(m_except);
    }
  }

  void TestServicesOnNonFrameworkThreadsAnalyzer::runOnOtherThread() {
    std::unique_lock<std::mutex> lk(m_mutex);

    do {
      m_condVar.wait(lk, [this] { return m_managerThreadReady; });
      if (m_continueProcessing) {
        edm::ModuleCallingContext newContext(*m_moduleCallingContext);
        edm::ModuleContextSentry sentry(&newContext, m_moduleCallingContext->parent());

        edm::ServiceRegistry::Operate srSentry(*m_serviceToken);
        try {
          edm::Service<edm::RandomNumberGenerator> rng;
          edm::Service<edm::MessageLogger> ml;
          ml->setThreadContext(*m_moduleCallingContext);
          edm::LogSystem("ModuleThread") << "  ++running with rng "
                                         << CLHEP::RandFlat::shootInt(&rng->getEngine(m_streamID), 10);
        } catch (...) {
          m_except = std::current_exception();
        }
      }
      m_eventWorkDone = true;
      m_managerThreadReady = false;
      lk.unlock();
      m_condVar.notify_one();
      lk.lock();
    } while (m_continueProcessing);
  }

  void TestServicesOnNonFrameworkThreadsAnalyzer::shutdownThread() {
    m_continueProcessing = false;
    m_mutex.unlock();
    m_condVar.notify_one();
    m_thread->join();
  }

}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::TestServicesOnNonFrameworkThreadsAnalyzer);
