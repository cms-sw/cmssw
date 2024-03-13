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

    void preGlobalBeginRun(edm::GlobalContext const&);
    void preGlobalEndRun(edm::GlobalContext const&);
    void preGlobalBeginLumi(edm::GlobalContext const&);
    void preGlobalEndLumi(edm::GlobalContext const&);

    void preStreamBeginLumi(edm::StreamContext const&);
    void postStreamBeginLumi(edm::StreamContext const&);
    void preStreamEndLumi(edm::StreamContext const&);
    void postStreamEndLumi(edm::StreamContext const&);

    void preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);

    unsigned int nPreStreamBeginLumi() const;
    unsigned int nPostStreamBeginLumi() const;
    unsigned int nPreStreamEndLumi() const;
    unsigned int nPostStreamEndLumi() const;

    unsigned int nPreModuleStreamBeginLumi() const;
    unsigned int nPostModuleStreamBeginLumi() const;
    unsigned int nPreModuleStreamEndLumi() const;
    unsigned int nPostModuleStreamEndLumi() const;

  private:
    bool verbose_;

    std::atomic<unsigned int> nPreStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPostStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPreStreamEndLumi_ = 0;
    std::atomic<unsigned int> nPostStreamEndLumi_ = 0;

    std::atomic<unsigned int> nPreModuleStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPostModuleStreamBeginLumi_ = 0;
    std::atomic<unsigned int> nPreModuleStreamEndLumi_ = 0;
    std::atomic<unsigned int> nPostModuleStreamEndLumi_ = 0;
  };
}  // namespace edmtest
#endif
