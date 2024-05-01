// -*- C++ -*-
#ifndef FWCore_Integration_TestServiceTwo_h
#define FWCore_Integration_TestServiceTwo_h
//
// Package:     FWCore/Integration
// Class  :     TestServiceTwo
//
// Implementation:
//     Service initially intended for testing behavior after exceptions.
//     ExceptionThrowingProducer uses this and is in the same test plugin
//     library and could be used to access the service if it was ever useful
//     for debugging issues related to begin/end transitions.
//
//     This is almost identical to TestServiceOne. It was initially used to
//     test that after a signal all services get executed even if one of the
//     service functions throws.
//
//
// Original Author:  W. David Dagenhart
//         Created:  13 March 2024

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"

#include <atomic>

namespace edmtest {

  class TestServiceTwo {
  public:
    TestServiceTwo(const edm::ParameterSet&, edm::ActivityRegistry&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

    void preBeginProcessBlock(edm::GlobalContext const&);
    void preEndProcessBlock(edm::GlobalContext const&);

    void preGlobalBeginRun(edm::GlobalContext const&);
    void preGlobalEndRun(edm::GlobalContext const&);

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
  };
}  // namespace edmtest
#endif
