
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTRangeT0;
class DTRangeT0Rcd;

class DTRangeT0ValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTRangeT0ValidateDBRead(edm::ParameterSet const& p);
  explicit DTRangeT0ValidateDBRead(int i);
  ~DTRangeT0ValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTRangeT0, DTRangeT0Rcd> dtrangeToken_;
};
