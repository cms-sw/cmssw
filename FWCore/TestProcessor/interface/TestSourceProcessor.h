#ifndef FWCore_TestProcessor_TestSourceProcessor_h
#define FWCore_TestProcessor_TestSourceProcessor_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     TestSourceProcessor
//
/**\class TestSourceProcessor TestSourceProcessor.h "TestSourceProcessor.h"

 Description: Used for testing InputSources

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 30 Apr 2018 18:51:00 GMT
//
#include <string>
#include <utility>
#include <memory>
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

#include "FWCore/Common/interface/FWCoreCommonFwd.h"

#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/PrincipalCache.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/MergeableRunProductProcesses.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "FWCore/TestProcessor/interface/EventFromSource.h"
#include "FWCore/TestProcessor/interface/LuminosityBlockFromSource.h"
#include "FWCore/TestProcessor/interface/ProcessBlock.h"
#include "FWCore/TestProcessor/interface/RunFromSource.h"

namespace edm::test {

  class TestSourceProcessor {
  public:
    TestSourceProcessor(std::string const& iConfig, ServiceToken iToken = ServiceToken());
    ~TestSourceProcessor();

    InputSource::ItemTypeInfo findNextTransition();

    std::shared_ptr<FileBlock> openFile();
    void closeFile(std::shared_ptr<FileBlock>);

    edm::test::RunFromSource readRun();

    edm::test::LuminosityBlockFromSource readLuminosityBlock();

    edm::test::EventFromSource readEvent();

  private:
    edm::InputSource::ItemTypeInfo lastTransition_;

    oneapi::tbb::global_control globalControl_;
    oneapi::tbb::task_group taskGroup_;
    oneapi::tbb::task_arena arena_;
    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    std::shared_ptr<ProductRegistry> preg_;
    std::shared_ptr<BranchIDListHelper> branchIDListHelper_;
    std::shared_ptr<ProcessBlockHelper> processBlockHelper_;
    std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper_;
    ServiceToken serviceToken_;

    std::shared_ptr<ProcessConfiguration const> processConfiguration_;
    ProcessContext processContext_;
    MergeableRunProductProcesses mergeableRunProductProcesses_;

    ProcessHistoryRegistry processHistoryRegistry_;
    std::unique_ptr<HistoryAppender> historyAppender_;

    PrincipalCache principalCache_;
    PreallocationConfiguration preallocations_;

    std::unique_ptr<edm::InputSource> source_;

    std::shared_ptr<RunPrincipal> runPrincipal_;
    std::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal_;

    std::shared_ptr<FileBlock> fb_;
  };
}  // namespace edm::test

#endif