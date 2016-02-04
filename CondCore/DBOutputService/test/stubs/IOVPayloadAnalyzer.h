#ifndef CondCore_DBOutputService_test_IOVPayloadAnalyzer_h
#define CondCore_DBOutputService_test_IOVPayloadAnalyzer_h
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

class IOVPayloadAnalyzer : public edm::EDAnalyzer {
 public:
  explicit IOVPayloadAnalyzer(const edm::ParameterSet& iConfig );
  virtual ~IOVPayloadAnalyzer();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();
 private:
  std::string m_record;
  // ----------member data ---------------------------
};
#endif
