#ifndef CondCore_DBOutputService_test_LumiBasedUpdateAnalyzer_h
#define CondCore_DBOutputService_test_LumiBasedUpdateAnalyzer_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include <chrono>
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

//
// class declaration
//

class LumiBasedUpdateAnalyzer : public edm::EDAnalyzer {
public:
  explicit LumiBasedUpdateAnalyzer(const edm::ParameterSet& iConfig);
  virtual ~LumiBasedUpdateAnalyzer();
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  std::string m_record;
  bool m_tagLocks;
  // ----------member data ---------------------------
};
#endif
