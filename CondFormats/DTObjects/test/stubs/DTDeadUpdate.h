
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

namespace edmtest {
  class DTDeadUpdate : public edm::one::EDAnalyzer<> {
  public:
    explicit DTDeadUpdate(edm::ParameterSet const& p);
    explicit DTDeadUpdate(int i);
    ~DTDeadUpdate() override;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void endJob() override;

  private:
    void fill_dead_HV(const char* file, DTDeadFlag* deadList);
    void fill_dead_TP(const char* file, DTDeadFlag* deadList);
    void fill_dead_RO(const char* file, DTDeadFlag* deadList);
    void fill_discCat(const char* file, DTDeadFlag* deadList);
    DTDeadFlag* dSum;
    edm::ESGetToken<DTDeadFlag, DTDeadFlagRcd> es_token;
  };
}  // namespace edmtest
