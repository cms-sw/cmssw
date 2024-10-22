
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTLVStatus;
class DTLVStatusRcd;
class DTLVStatusValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTLVStatusValidateDBRead(edm::ParameterSet const& p);
  explicit DTLVStatusValidateDBRead(int i);
  ~DTLVStatusValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTLVStatus, DTLVStatusRcd> dtlvstatusToken_;
};
