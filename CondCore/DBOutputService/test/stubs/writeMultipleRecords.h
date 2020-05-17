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

class writeMultipleRecords : public edm::EDAnalyzer {
public:
  explicit writeMultipleRecords(const edm::ParameterSet& iConfig);
  ~writeMultipleRecords() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

private:
  std::string m_PedRecordName;
  std::string m_StripRecordName;
};
