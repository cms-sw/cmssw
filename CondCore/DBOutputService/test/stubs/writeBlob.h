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

class writeBlob : public edm::EDAnalyzer {
public:
  explicit writeBlob(const edm::ParameterSet& iConfig);
  ~writeBlob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

private:
  std::string m_StripRecordName;
};
