#ifndef CondCore_DBOutputService_test_Timestamp_h
#define CondCore_DBOutputService_test_Timestamp_h
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

class  Timestamp: public edm::EDAnalyzer {
 public:
  explicit Timestamp(const edm::ParameterSet& iConfig );
  virtual ~Timestamp();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();
 private:
  std::string m_record;
  // ----------member data ---------------------------
};
#endif
