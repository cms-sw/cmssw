
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
class DTDeadFlag;
class DTDeadFlagRcd;
class DTDeadFlagValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTDeadFlagValidateDBRead(edm::ParameterSet const& p);
  explicit DTDeadFlagValidateDBRead(int i);
  ~DTDeadFlagValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTDeadFlag, DTDeadFlagRcd> dtdeadflagToken_;
};
