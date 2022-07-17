
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
class DTHVStatus;
class DTHVStatusRcd;
class DTHVStatusValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTHVStatusValidateDBRead(edm::ParameterSet const& p);
  explicit DTHVStatusValidateDBRead(int i);
  ~DTHVStatusValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTHVStatus, DTHVStatusRcd> dthvstatusToken_;
};
