
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTT0;
class DTT0Rcd;
class DTT0ValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTT0ValidateDBRead(edm::ParameterSet const& p);
  explicit DTT0ValidateDBRead(int i);
  ~DTT0ValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTT0, DTT0Rcd> dtT0Token_;
};
