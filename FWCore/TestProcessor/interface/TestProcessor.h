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
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"


#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "FWCore/TestProcessor/interface/Event.h"

#include "FWCore/Utilities/interface/EDPutToken.h"

// forward declarations

namespace edm {
  class ThinnedAssociationsHelper;
  class ExceptionToActionTable;
  class HistoryAppender;

  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupsController;
  }

namespace test {

class TestProcessorConfig {
  public:
    TestProcessorConfig(std::string const& iPythonConfiguration):
    config_(iPythonConfiguration)
    {}
    std::string const& pythonConfiguration() const { return config_;}
    void setPythonConfiguration(std::string iConfig) { config_ = std::move(iConfig);}
    
    /** add a Process name to the Process history. If multiple calls are made to addProcess,
     then the call order will be the order of the Processes in the history.*/
    void addExtraProcess(std::string const& iProcessName);
    
    std::vector<std::string> const& extraProcesses() const { return extraProcesses_;}
    
    /** A blank iProcessName means it produces in the primary process.
     The use of any other name requires that it is first added via `addExtraProcess`.
     */
    template<typename T>
    edm::EDPutTokenT<T> produces(std::string iModuleLabel,
                                 std::string iProductInstanceLabel = std::string(),
                                 std::string iProcessName = std::string()) {
      produceEntries_.emplace_back(edm::TypeID(typeid(T)), std::move(iModuleLabel),
                                   std::move(iProductInstanceLabel),
                                   std::move(iProcessName));
      return edm::EDPutTokenT<T>(produceEntries_.size()-1);
    }
    
    struct ProduceEntry {
      ProduceEntry(edm::TypeID const& iType,
                   std::string moduleLabel,
                   std::string instanceLabel,
                   std::string processName):
      type_{iType},
      moduleLabel_{std::move(moduleLabel)},
      instanceLabel_{std::move(instanceLabel)},
      processName_{std::move(processName)} {}
      edm::TypeID type_;
      std::string moduleLabel_;
      std::string instanceLabel_;
      std::string processName_;
    };
  
  std::vector<ProduceEntry> const& produceEntries() const { return produceEntries_;}
  private:
    std::string config_;
    std::vector<std::string> extraProcesses_;
    std::vector<ProduceEntry> produceEntries_;
  };

class TestProcessor
{
  public:
  using Config = TestProcessorConfig;
  
  TestProcessor(Config const& iConfig);
  ~TestProcessor() noexcept(false);

  /** Run the test. The function arguments are the data products to be added to the
     Event for this particular test.
     */
  template< typename... T>
  edm::test::Event test(T&&... iArgs) {
    return testImpl(std::forward<T>(iArgs)...);
  }
 
  void setRunNumber(edm::RunNumber_t);
  void setLuminosityBlockNumber(edm::LuminosityBlockNumber_t);
  
  std::string const& labelOfTestModule() const {
    return labelOfTestModule_;
  }
  
  private:
  TestProcessor(const TestProcessor&) = delete; // stop default

  const TestProcessor& operator=(const TestProcessor&) = delete; // stop default

  template< typename T, typename... U>
  edm::test::Event testImpl(std::pair<edm::EDPutTokenT<T>,std::unique_ptr<T>>&& iPut, U&&... iArgs) {
    put(std::move(iPut));
    return testImpl(std::forward(iArgs)...);
  }
  
  template< typename T>
  void put(std::pair<edm::EDPutTokenT<T>,std::unique_ptr<T>>&& iPut) {
    put(iPut.first.index(), std::make_unique<edm::Wrapper<T>>(std::move(iPut.second)));
  }
  
  void put(unsigned int, std::unique_ptr<WrapperBase>);
  
  edm::test::Event testImpl();
  
  void setupProcessing();
  void teardownProcessing();
  
  void beginJob();
  void beginRun();
  void beginLuminosityBlock();
  void event();
  void endLuminosityBlock();
  void endRun();
  void endJob();
  
  // ---------- member data --------------------------------
  std::string labelOfTestModule_;
  std::shared_ptr<ActivityRegistry> actReg_; // We do not use propagate_const because the registry itself is mutable.
  std::shared_ptr<ProductRegistry> preg_;
  std::shared_ptr<BranchIDListHelper> branchIDListHelper_;
  std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper_;
  ServiceToken serviceToken_;
  std::unique_ptr<eventsetup::EventSetupsController> espController_;
  std::shared_ptr<eventsetup::EventSetupProvider> esp_;
  
  std::unique_ptr<ExceptionToActionTable const>          act_table_;
  std::shared_ptr<ProcessConfiguration const>       processConfiguration_;
  ProcessContext                                processContext_;

  ProcessHistoryRegistry processHistoryRegistry_;
  std::unique_ptr<HistoryAppender> historyAppender_;

  PrincipalCache                                principalCache_;
  PreallocationConfiguration                    preallocations_;

  std::shared_ptr<ModuleRegistry> moduleRegistry_;
  std::unique_ptr<Schedule> schedule_;
  std::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal_;
  
  std::vector<std::pair<edm::BranchDescription,std::unique_ptr<WrapperBase>>> dataProducts_;
  
  RunNumber_t runNumber_=1;
  LuminosityBlockNumber_t lumiNumber_=1;
  EventNumber_t eventNumber_=1;
  bool beginJobCalled_ = false;
  bool beginRunCalled_ = false;
  bool beginLumiCalled_ = false;
  bool newRun_ = true;
  bool newLumi_ = true;
};
}
}

#endif
