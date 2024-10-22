
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTReadOutMapping;
class DTReadOutMappingRcd;
class DTCompMapValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTCompMapValidateDBRead(edm::ParameterSet const& p);
  explicit DTCompMapValidateDBRead(int i);
  ~DTCompMapValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTReadOutMapping, DTReadOutMappingRcd> dtreadoutmappingToken_;
};
