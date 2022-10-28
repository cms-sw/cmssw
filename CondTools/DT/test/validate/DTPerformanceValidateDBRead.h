
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTPerformance;
class DTPerformanceRcd;
class DTPerformanceValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTPerformanceValidateDBRead(edm::ParameterSet const& p);
  explicit DTPerformanceValidateDBRead(int i);
  ~DTPerformanceValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTPerformance, DTPerformanceRcd> dtperfToken_;
};
