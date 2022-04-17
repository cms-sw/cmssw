
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTStatusFlag;
class DTStatusFlagRcd;
class DTStatusFlagValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTStatusFlagValidateDBRead(edm::ParameterSet const& p);
  explicit DTStatusFlagValidateDBRead(int i);
  ~DTStatusFlagValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTStatusFlag, DTStatusFlagRcd> dtstatusFlagToken_;
};
