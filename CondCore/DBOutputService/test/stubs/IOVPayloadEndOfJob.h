#ifndef CondCore_DBOutputService_test_IOVPayloadEndOfJob_h
#define CondCore_DBOutputService_test_IOVPayloadEndOfJob_h
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
class Pedestals;
class IOVPayloadEndOfJob : public edm::EDAnalyzer {
 public:
  explicit IOVPayloadEndOfJob(const edm::ParameterSet& iConfig );
  virtual ~IOVPayloadEndOfJob();
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();
 private:
  std::string m_record;
};
#endif
