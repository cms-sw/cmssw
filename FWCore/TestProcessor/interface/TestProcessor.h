#ifndef FWCore_TestProcessor_TestProcessor_h
#define FWCore_TestProcessor_TestProcessor_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     TestProcessor
//
/**\class TestProcessor TestProcessor.h "TestProcessor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 30 Apr 2018 18:51:00 GMT
//

// system include files
#include <string>
#include <utility>
#include <memory>

// user include files
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/ModuleRegistry.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "FWCore/TestProcessor/interface/Event.h"
#include "FWCore/TestProcessor/interface/LuminosityBlock.h"
#include "FWCore/TestProcessor/interface/Run.h"
#include "FWCore/TestProcessor/interface/TestDataProxy.h"
#include "FWCore/TestProcessor/interface/ESPutTokenT.h"
#include "FWCore/TestProcessor/interface/ESProduceEntry.h"
#include "FWCore/TestProcessor/interface/EventSetupTestHelper.h"

#include "FWCore/Utilities/interface/EDPutToken.h"

// forward declarations

namespace edm {
  class ThinnedAssociationsHelper;
  class ExceptionToActionTable;
  class HistoryAppender;

  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupsController;
  }  // namespace eventsetup

  namespace test {
    class TestProcessorConfig;
    class EventSetupTestHelper;

    class ProcessToken {
      friend TestProcessorConfig;

    public:
      ProcessToken() : index_{undefinedIndex()} {}

      int index() const { return index_; }

      static int undefinedIndex() { return -1; }

    private:
      explicit ProcessToken(int index) : index_{index} {}

      int index_;
    };

    class TestProcessorConfig {
    public:
      TestProcessorConfig(std::string const& iPythonConfiguration) : config_(iPythonConfiguration) {}
      std::string const& pythonConfiguration() const { return config_; }
      void setPythonConfiguration(std::string iConfig) { config_ = std::move(iConfig); }

      /** add a Process name to the Process history. If multiple calls are made to addProcess,
     then the call order will be the order of the Processes in the history.*/
      ProcessToken addExtraProcess(std::string const& iProcessName) {
        extraProcesses_.emplace_back(iProcessName);
        return ProcessToken(extraProcesses_.size() - 1);
      }

      std::vector<std::string> const& extraProcesses() const { return extraProcesses_; }

      /** A blank iProcessName means it produces in the primary process.
     The use of any other name requires that it is first added via `addExtraProcess`.
     */
      template <typename T>
      edm::EDPutTokenT<T> produces(std::string iModuleLabel,
                                   std::string iProductInstanceLabel = std::string(),
                                   ProcessToken iToken = ProcessToken()) {
        produceEntries_.emplace_back(
            edm::TypeID(typeid(T)), std::move(iModuleLabel), std::move(iProductInstanceLabel), processName(iToken));
        return edm::EDPutTokenT<T>(produceEntries_.size() - 1);
      }

      template <typename REC, typename T>
      edm::test::ESPutTokenT<T> esProduces(std::string iLabel = std::string()) {
        auto rk = eventsetup::EventSetupRecordKey::makeKey<REC>();
        eventsetup::DataKey dk(eventsetup::DataKey::makeTypeTag<T>(), iLabel.c_str());
        esProduceEntries_.emplace_back(rk, dk, std::make_shared<TestDataProxy<T>>());
        return edm::test::ESPutTokenT<T>(esProduceEntries_.size() - 1);
      }

      struct ProduceEntry {
        ProduceEntry(edm::TypeID const& iType,
                     std::string moduleLabel,
                     std::string instanceLabel,
                     std::string processName)
            : type_{iType},
              moduleLabel_{std::move(moduleLabel)},
              instanceLabel_{std::move(instanceLabel)},
              processName_{std::move(processName)} {}
        edm::TypeID type_;
        std::string moduleLabel_;
        std::string instanceLabel_;
        std::string processName_;
      };

      std::vector<ProduceEntry> const& produceEntries() const { return produceEntries_; }

      std::vector<ESProduceEntry> const& esProduceEntries() const { return esProduceEntries_; }

    private:
      std::string config_;
      std::vector<std::string> extraProcesses_;
      std::vector<ProduceEntry> produceEntries_;
      std::vector<ESProduceEntry> esProduceEntries_;

      std::string processName(ProcessToken iToken) {
        if (iToken.index() == ProcessToken::undefinedIndex()) {
          return std::string();
        }
        return extraProcesses_[iToken.index()];
      }
    };

    class TestProcessor {
    public:
      using Config = TestProcessorConfig;

      TestProcessor(Config const& iConfig, ServiceToken iToken = ServiceToken());
      ~TestProcessor() noexcept(false);

      /** Run the test. The function arguments are the data products to be added to the
     Event for this particular test.
     */
      template <typename... T>
      edm::test::Event test(T&&... iArgs) {
        return testImpl(std::forward<T>(iArgs)...);
      }

      template <typename... T>
      edm::test::LuminosityBlock testBeginLuminosityBlock(edm::LuminosityBlockNumber_t iNum, T&&... iArgs) {
        return testBeginLuminosityBlockImpl(iNum, std::forward<T>(iArgs)...);
      }
      template <typename... T>
      edm::test::LuminosityBlock testEndLuminosityBlock(T&&... iArgs) {
        return testEndLuminosityBlockImpl(std::forward<T>(iArgs)...);
      }

      template <typename... T>
      edm::test::Run testBeginRun(edm::RunNumber_t iNum, T&&... iArgs) {
        return testBeginRunImpl(iNum, std::forward<T>(iArgs)...);
      }
      template <typename... T>
      edm::test::Run testEndRun(T&&... iArgs) {
        return testEndRunImpl(std::forward<T>(iArgs)...);
      }

      /** Run only beginJob and endJob. Once this is used, you should not attempt to run any further tests.
This simulates a problem happening early in the job which causes processing not to proceed.
   */
      void testBeginAndEndJobOnly() {
        beginJob();
        endJob();
      }

      void testRunWithNoLuminosityBlocks() {
        beginJob();
        beginRun();
        endRun();
        endJob();
      }

      void testLuminosityBlockWithNoEvents() {
        beginJob();
        beginRun();
        beginLuminosityBlock();
        endLuminosityBlock();
        endRun();
        endJob();
      }
      void setRunNumber(edm::RunNumber_t);
      void setLuminosityBlockNumber(edm::LuminosityBlockNumber_t);
      void setEventNumber(edm::EventNumber_t);

      std::string const& labelOfTestModule() const { return labelOfTestModule_; }

    private:
      TestProcessor(const TestProcessor&) = delete;  // stop default

      const TestProcessor& operator=(const TestProcessor&) = delete;  // stop default

      template <typename T, typename... U>
      edm::test::Event testImpl(std::pair<edm::EDPutTokenT<T>, std::unique_ptr<T>>&& iPut, U&&... iArgs) {
        put(std::move(iPut));
        return testImpl(std::forward<U>(iArgs)...);
      }

      template <typename T, typename... U>
      edm::test::Event testImpl(std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>>&& iPut, U&&... iArgs) {
        put(std::move(iPut));
        return testImpl(std::forward<U>(iArgs)...);
      }

      template <typename T>
      void put(std::pair<edm::EDPutTokenT<T>, std::unique_ptr<T>>&& iPut) {
        put(iPut.first.index(), std::make_unique<edm::Wrapper<T>>(std::move(iPut.second)));
      }

      template <typename T>
      void put(std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>>&& iPut) {
        dynamic_cast<TestDataProxy<T>*>(esHelper_->getProxy(iPut.first.index()).get())->setData(std::move(iPut.second));
      }

      void put(unsigned int, std::unique_ptr<WrapperBase>);

      edm::test::Event testImpl();

      template <typename T, typename... U>
      edm::test::LuminosityBlock testBeginLuminosityBlockImpl(
          edm::LuminosityBlockNumber_t iNum,
          std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>>&& iPut,
          U&&... iArgs) {
        put(std::move(iPut));
        return testBeginLuminosityBlockImpl(iNum, std::forward<U>(iArgs)...);
      }
      edm::test::LuminosityBlock testBeginLuminosityBlockImpl(edm::LuminosityBlockNumber_t);

      template <typename T, typename... U>
      edm::test::LuminosityBlock testEndLuminosityBlockImpl(
          std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>>&& iPut, U&&... iArgs) {
        put(std::move(iPut));
        return testEndLuminosityBlockImpl(std::forward<U>(iArgs)...);
      }
      edm::test::LuminosityBlock testEndLuminosityBlockImpl();

      template <typename T, typename... U>
      edm::test::Run testBeginRunImpl(edm::RunNumber_t iNum,
                                      std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>>&& iPut,
                                      U&&... iArgs) {
        put(std::move(iPut));
        return testBeginRunImpl(iNum, std::forward<U>(iArgs)...);
      }
      edm::test::Run testBeginRunImpl(edm::RunNumber_t);
      template <typename T, typename... U>
      edm::test::LuminosityBlock testEndRunImpl(std::pair<edm::test::ESPutTokenT<T>, std::unique_ptr<T>>&& iPut,
                                                U&&... iArgs) {
        put(std::move(iPut));
        return testEndRunImpl(std::forward<U>(iArgs)...);
      }
      edm::test::Run testEndRunImpl();

      void setupProcessing();
      void teardownProcessing();

      void beginJob();
      void beginRun();
      void beginLuminosityBlock();
      void event();
      std::shared_ptr<LuminosityBlockPrincipal> endLuminosityBlock();
      std::shared_ptr<RunPrincipal> endRun();
      void endJob();

      // ---------- member data --------------------------------
      std::string labelOfTestModule_;
      std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
      std::shared_ptr<ProductRegistry> preg_;
      std::shared_ptr<BranchIDListHelper> branchIDListHelper_;
      std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper_;
      ServiceToken serviceToken_;
      std::unique_ptr<eventsetup::EventSetupsController> espController_;
      std::shared_ptr<eventsetup::EventSetupProvider> esp_;
      std::shared_ptr<EventSetupTestHelper> esHelper_;

      std::unique_ptr<ExceptionToActionTable const> act_table_;
      std::shared_ptr<ProcessConfiguration const> processConfiguration_;
      ProcessContext processContext_;

      ProcessHistoryRegistry processHistoryRegistry_;
      std::unique_ptr<HistoryAppender> historyAppender_;

      PrincipalCache principalCache_;
      PreallocationConfiguration preallocations_;

      std::shared_ptr<ModuleRegistry> moduleRegistry_;
      std::unique_ptr<Schedule> schedule_;
      std::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal_;

      std::vector<std::pair<edm::BranchDescription, std::unique_ptr<WrapperBase>>> dataProducts_;

      RunNumber_t runNumber_ = 1;
      LuminosityBlockNumber_t lumiNumber_ = 1;
      EventNumber_t eventNumber_ = 1;
      bool beginJobCalled_ = false;
      bool beginRunCalled_ = false;
      bool beginLumiCalled_ = false;
    };
  }  // namespace test
}  // namespace edm

#endif
