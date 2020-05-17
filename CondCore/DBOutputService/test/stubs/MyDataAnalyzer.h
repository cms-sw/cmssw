#ifndef CondCore_DBOutputService_test_MyDataAnalyzer_h
#define CondCore_DBOutputService_test_MyDataAnalyzer_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

//
// class decleration
//
class MyDataAnalyzer : public edm::EDAnalyzer {
public:
  explicit MyDataAnalyzer(const edm::ParameterSet& iConfig);
  ~MyDataAnalyzer() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;

private:
  std::string m_record;
  bool m_LoggingOn;
};
#endif
