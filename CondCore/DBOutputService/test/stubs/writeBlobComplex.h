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

class writeBlobComplex : public edm::EDAnalyzer {
public:
  explicit writeBlobComplex(const edm::ParameterSet& iConfig);
  ~writeBlobComplex() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

private:
  std::string m_RecordName;
};
