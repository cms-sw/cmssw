#ifndef CondCore_DBOutputService_test_IOVPayloadEndOfJob_h
#define CondCore_DBOutputService_test_IOVPayloadEndOfJob_h
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
class Pedestals;
class IOVPayloadEndOfJob : public edm::EDAnalyzer {
public:
  explicit IOVPayloadEndOfJob(const edm::ParameterSet& iConfig);
  ~IOVPayloadEndOfJob() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;

private:
  std::string m_record;
};
#endif
