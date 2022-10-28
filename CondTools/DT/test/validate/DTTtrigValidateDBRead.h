
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
class DTTtrig;
class DTTtrigRcd;

class DTTtrigValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTTtrigValidateDBRead(edm::ParameterSet const& p);
  explicit DTTtrigValidateDBRead(int i);
  ~DTTtrigValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTTtrig, DTTtrigRcd> dtTrigToken_;
};
