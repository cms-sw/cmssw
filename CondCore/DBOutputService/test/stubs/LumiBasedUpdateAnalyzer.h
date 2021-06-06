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
  virtual void beginJob();
  virtual void endJob();
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

private:
  std::string m_record;
  int m_ret;
  // ----------member data ---------------------------
};
#endif
