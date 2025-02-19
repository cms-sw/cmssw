#ifndef CondCore_DBOutputService_test_MyDataAnalyzer_h
#define CondCore_DBOutputService_test_MyDataAnalyzer_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
namespace edm{
  class ParameterSet;
  class Event;
  class EventSetup;
}

//
// class decleration
//
class MyDataAnalyzer : public edm::EDAnalyzer {
 public:
  explicit MyDataAnalyzer(const edm::ParameterSet& iConfig );
  virtual ~MyDataAnalyzer();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();
 private:
  std::string m_record;
  bool m_LoggingOn;
};
#endif
