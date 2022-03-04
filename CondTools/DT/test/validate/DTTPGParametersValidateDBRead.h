
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTTPGParameters;
class DTTPGParametersRcd;
class DTTPGParametersValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTTPGParametersValidateDBRead(edm::ParameterSet const& p);
  explicit DTTPGParametersValidateDBRead(int i);
  ~DTTPGParametersValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTTPGParameters, DTTPGParametersRcd> dttpgPramToken_;
};
