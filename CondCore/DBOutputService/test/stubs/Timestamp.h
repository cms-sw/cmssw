#ifndef CondCore_DBOutputService_test_Timestamp_h
#define CondCore_DBOutputService_test_Timestamp_h
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

class Timestamp : public edm::EDAnalyzer {
public:
  explicit Timestamp(const edm::ParameterSet& iConfig);
  ~Timestamp() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;

private:
  std::string m_record;
  // ----------member data ---------------------------
};
#endif
