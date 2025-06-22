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

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"

#include <atomic>

namespace edmtest {

  class TestServiceOne {
  public:
    TestServiceOne(const edm::ParameterSet&, edm::ActivityRegistry&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

    void preBeginJob(edm::ProcessContext const&);
    void postBeginJob();
    void preEndJob();
    void postEndJob();

    void preModuleBeginJob(edm::ModuleDescription const&);
    void postModuleBeginJob(edm::ModuleDescription const&);
    void preModuleEndJob(edm::ModuleDescription const&);
    void postModuleEndJob(edm::ModuleDescription const&);

    void preBeginStream(edm::StreamContext const&);
    void postBeginStream(edm::StreamContext const&);
    void preEndStream(edm::StreamContext const&);
    void postEndStream(edm::StreamContext const&);

    void preModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void preModuleEndStream(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleEndStream(edm::StreamContext const&, edm::ModuleCallingContext const&);

    void preBeginProcessBlock(edm::GlobalContext const&);
    void postBeginProcessBlock(edm::GlobalContext const&);
    void preEndProcessBlock(edm::GlobalContext const&);
    void postEndProcessBlock(edm::GlobalContext const&);

    void preModuleBeginProcessBlock(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void postModuleBeginProcessBlock(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void preModuleEndProcessBlock(edm::GlobalContext const&, edm::ModuleCallingContext const&);
    void postModuleEndProcessBlock(edm::GlobalContext const&, edm::ModuleCallingContext const&);

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

    unsigned int nPreBeginJob() const;
    unsigned int nPostBeginJob() const;
    unsigned int nPreEndJob() const;
    unsigned int nPostEndJob() const;

    unsigned int nPreModuleBeginJob() const;
    unsigned int nPostModuleBeginJob() const;
    unsigned int nPreModuleEndJob() const;
    unsigned int nPostModuleEndJob() const;

    unsigned int nPreBeginStream() const;
    unsigned int nPostBeginStream() const;
    unsigned int nPreEndStream() const;
    unsigned int nPostEndStream() const;

    unsigned int nPreModuleBeginStream() const;
    unsigned int nPostModuleBeginStream() const;
    unsigned int nPreModuleEndStream() const;
    unsigned int nPostModuleEndStream() const;

    unsigned int nPreBeginProcessBlock() const;
    unsigned int nPostBeginProcessBlock() const;
    unsigned int nPreEndProcessBlock() const;
    unsigned int nPostEndProcessBlock() const;

    unsigned int nPreModuleBeginProcessBlock() const;
    unsigned int nPostModuleBeginProcessBlock() const;
    unsigned int nPreModuleEndProcessBlock() const;
    unsigned int nPostModuleEndProcessBlock() const;

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

    std::atomic<unsigned int> nPreBeginJob_ = 0;
    std::atomic<unsigned int> nPostBeginJob_ = 0;
    std::atomic<unsigned int> nPreEndJob_ = 0;
    std::atomic<unsigned int> nPostEndJob_ = 0;

    std::atomic<unsigned int> nPreModuleBeginJob_ = 0;
    std::atomic<unsigned int> nPostModuleBeginJob_ = 0;
    std::atomic<unsigned int> nPreModuleEndJob_ = 0;
    std::atomic<unsigned int> nPostModuleEndJob_ = 0;

    std::atomic<unsigned int> nPreBeginStream_ = 0;
    std::atomic<unsigned int> nPostBeginStream_ = 0;
    std::atomic<unsigned int> nPreEndStream_ = 0;
    std::atomic<unsigned int> nPostEndStream_ = 0;

    std::atomic<unsigned int> nPreModuleBeginStream_ = 0;
    std::atomic<unsigned int> nPostModuleBeginStream_ = 0;
    std::atomic<unsigned int> nPreModuleEndStream_ = 0;
    std::atomic<unsigned int> nPostModuleEndStream_ = 0;

    std::atomic<unsigned int> nPreBeginProcessBlock_ = 0;
    std::atomic<unsigned int> nPostBeginProcessBlock_ = 0;
    std::atomic<unsigned int> nPreEndProcessBlock_ = 0;
    std::atomic<unsigned int> nPostEndProcessBlock_ = 0;

    std::atomic<unsigned int> nPreModuleBeginProcessBlock_ = 0;
    std::atomic<unsigned int> nPostModuleBeginProcessBlock_ = 0;
    std::atomic<unsigned int> nPreModuleEndProcessBlock_ = 0;
    std::atomic<unsigned int> nPostModuleEndProcessBlock_ = 0;

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
