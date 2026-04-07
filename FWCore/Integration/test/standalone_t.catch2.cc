/*----------------------------------------------------------------------

Test of a non cmsRun executable

Note that the commented out lines are what is necessary to
setup the MessageLogger in this test. Note that tests like
this will hang after 1000 messages are sent to the MessageLogger
if the MessageLogger is not runnning.

----------------------------------------------------------------------*/

#include "catch2/catch_all.hpp"

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
// #include "FWCore/Utilities/interface/Presence.h"
// #include "FWCore/PluginManager/interface/PresenceFactory.h"

#include <memory>
#include <string>

TEST_CASE("Standalone", "[Integration]") {
  edm::propagate_const<std::unique_ptr<edm::AssertHandler>> m_handler;
  edm::propagate_const<std::unique_ptr<edm::ThreadsController>> m_scheduler;

  m_handler = std::make_unique<edm::AssertHandler>();
  m_scheduler = std::make_unique<edm::ThreadsController>(1);

  SECTION("writeAndReadFile") {
    {
      std::string configuration(
          "import FWCore.ParameterSet.Config as cms\n"
          "process = cms.Process('TEST')\n"
          "process.maxEvents = cms.untracked.PSet(\n"
          "    input = cms.untracked.int32(5)\n"
          ")\n"
          "process.source = cms.Source('EmptySource')\n"
          "process.JobReportService = cms.Service('JobReportService')\n"
          "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
          "process.m1 = cms.EDProducer('IntProducer',\n"
          "    ivalue = cms.int32(11)\n"
          ")\n"
          "process.out = cms.OutputModule('AsciiOutputModule')\n"
          "process.p = cms.Path(process.m1)\n"
          "process.e = cms.EndPath(process.out)\n");

      edm::EventProcessor proc(edm::getPSetFromConfig(configuration));
      proc.beginJob();
      proc.run();
      proc.endJob();
    }

    m_handler = nullptr;  // propagate_const<T> has no reset() function
  }
}
