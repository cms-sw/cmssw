// -*- C++ -*-
#ifndef FWCore_Integration_TestServiceOne_h
#define FWCore_Integration_TestServiceOne_h
//
// Package:     FWCore/Integration
// Class  :     TestServiceOne
//
// Implementation:
//     Service initially intended for testing behavior after exceptions.
//     ExceptionThrowingProducer uses this and is in the same test plugin
//     library and could be used to access the service if it was ever useful
//     for debugging issues related to begin/end transitions.
//
// Original Author:  W. David Dagenhart
//         Created:  13 March 2024

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"

#include <atomic>

namespace edmtest {

  class TestServiceOne {
  public:
    TestServiceOne(const edm::ParameterSet&, edm::ActivityRegistry&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

    void preBeginProcessBlock(edm::GlobalContext const&);
    void preEndProcessBlock(edm::GlobalContext const&);

    void preStreamBeginLumi(edm::StreamContext const&);
    void postStreamBeginLumi(edm::StreamContext const&);
    void preStreamEndLumi(edm::StreamContext const&);
    void postStreamEndLumi(edm::StreamContext const&);

    void preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);

    void preGlobalBeginLumi(edm::GlobalContext const&);
    void postGlobalBeginLumi(edm::GlobalContext const&);
    void preGlobalEndLumi(edm::GlobalContext const&);
    void postGlobalEndLumi(edm::GlobalContext const&);

    void preModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void postModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void preModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void postModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);

    void preGlobalWriteLumi(edm::GlobalContext const&);
    void postGlobalWriteLumi(edm::GlobalContext const&);

    void preStreamBeginRun(edm::StreamContext const&);
    void postStreamBeginRun(edm::StreamContext const&);
    void preStreamEndRun(edm::StreamContext const&);
    void postStreamEndRun(edm::StreamContext const&);

    void preModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void preModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&);

    void preGlobalBeginRun(edm::GlobalContext const&);
    void postGlobalBeginRun(edm::GlobalContext const&);
    void preGlobalEndRun(edm::GlobalContext const&);
    void postGlobalEndRun(edm::GlobalContext const&);

    void preModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void postModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void preModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void postModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);

    void preGlobalWriteRun(edm::GlobalContext const&);
    void postGlobalWriteRun(edm::GlobalContext const&);

    unsigned int nPreStreamBeginLumi() const;
    unsigned int nPostStreamBeginLumi() const;
    unsigned int nPreStreamEndLumi() const;
    unsigned int nPostStreamEndLumi() const;

    unsigned int nPreModuleStreamBeginLumi() const;
    unsigned int nPostModuleStreamBeginLumi() const;
    unsigned int nPreModuleStreamEndLumi() const;
    unsigned int nPostModuleStreamEndLumi() const;

    unsigned int nPreGlobalBeginLumi() const;
    unsigned int nPostGlobalBeginLumi() const;
    unsigned int nPreGlobalEndLumi() const;
    unsigned int nPostGlobalEndLumi() const;

    unsigned int nPreModuleGlobalBeginLumi() const;
    unsigned int nPostModuleGlobalBeginLumi() const;
    unsigned int nPreModuleGlobalEndLumi() const;
    unsigned int nPostModuleGlobalEndLumi() const;

    unsigned int nPreGlobalWriteLumi() const;
    unsigned int nPostGlobalWriteLumi() const;

    unsigned int nPreStreamBeginRun() const;
    unsigned int nPostStreamBeginRun() const;
    unsigned int nPreStreamEndRun() const;
    unsigned int nPostStreamEndRun() const;

    unsigned int nPreModuleStreamBeginRun() const;
    unsigned int nPostModuleStreamBeginRun() const;
    unsigned int nPreModuleStreamEndRun() const;
    unsigned int nPostModuleStreamEndRun() const;

    unsigned int nPreGlobalBeginRun() const;
    unsigned int nPostGlobalBeginRun() const;
    unsigned int nPreGlobalEndRun() const;
    unsigned int nPostGlobalEndRun() const;

    unsigned int nPreModuleGlobalBeginRun() const;
    unsigned int nPostModuleGlobalBeginRun() const;
    unsigned int nPreModuleGlobalEndRun() const;
    unsigned int nPostModuleGlobalEndRun() const;

    unsigned int nPreGlobalWriteRun() const;
    unsigned int nPostGlobalWriteRun() const;

  private:
    bool verbose_;
    bool printTimestamps_;

    std::atomic<unsigned int> nPreStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPostStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPreStreamEndLumi_ = 0;
    std::atomic<unsigned int> nPostStreamEndLumi_ = 0;

    std::atomic<unsigned int> nPreModuleStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPostModuleStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPreModuleStreamEndLumi_ = 0;
    std::atomic<unsigned int> nPostModuleStreamEndLumi_ = 0;

    std::atomic<unsigned int> nPreGlobalBeginLumi_ = 0;
    std::atomic<unsigned int> nPostGlobalBeginLumi_ = 0;
    std::atomic<unsigned int> nPreGlobalEndLumi_ = 0;
    std::atomic<unsigned int> nPostGlobalEndLumi_ = 0;

    std::atomic<unsigned int> nPreModuleGlobalBeginLumi_ = 0;
    std::atomic<unsigned int> nPostModuleGlobalBeginLumi_ = 0;
    std::atomic<unsigned int> nPreModuleGlobalEndLumi_ = 0;
    std::atomic<unsigned int> nPostModuleGlobalEndLumi_ = 0;

    std::atomic<unsigned int> nPreGlobalWriteLumi_ = 0;
    std::atomic<unsigned int> nPostGlobalWriteLumi_ = 0;

    std::atomic<unsigned int> nPreStreamBeginRun_ = 0;
    std::atomic<unsigned int> nPostStreamBeginRun_ = 0;
    std::atomic<unsigned int> nPreStreamEndRun_ = 0;
    std::atomic<unsigned int> nPostStreamEndRun_ = 0;

    std::atomic<unsigned int> nPreModuleStreamBeginRun_ = 0;
    std::atomic<unsigned int> nPostModuleStreamBeginRun_ = 0;
    std::atomic<unsigned int> nPreModuleStreamEndRun_ = 0;
    std::atomic<unsigned int> nPostModuleStreamEndRun_ = 0;

    std::atomic<unsigned int> nPreGlobalBeginRun_ = 0;
    std::atomic<unsigned int> nPostGlobalBeginRun_ = 0;
    std::atomic<unsigned int> nPreGlobalEndRun_ = 0;
    std::atomic<unsigned int> nPostGlobalEndRun_ = 0;

    std::atomic<unsigned int> nPreModuleGlobalBeginRun_ = 0;
    std::atomic<unsigned int> nPostModuleGlobalBeginRun_ = 0;
    std::atomic<unsigned int> nPreModuleGlobalEndRun_ = 0;
    std::atomic<unsigned int> nPostModuleGlobalEndRun_ = 0;

    std::atomic<unsigned int> nPreGlobalWriteRun_ = 0;
    std::atomic<unsigned int> nPostGlobalWriteRun_ = 0;
  };
}  // namespace edmtest
#endif
