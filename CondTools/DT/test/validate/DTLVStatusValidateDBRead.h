
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTLVStatus;
class DTLVStatusRcd;
class DTLVStatusValidateDBRead : public edm::EDAnalyzer {
public:
  explicit DTLVStatusValidateDBRead(edm::ParameterSet const& p);
  explicit DTLVStatusValidateDBRead(int i);
  virtual ~DTLVStatusValidateDBRead();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob();

private:
  std::string dataFileName;
  std::string elogFileName;
  edm::ESGetToken<DTLVStatus, DTLVStatusRcd> dtlvstatusToken_;
};
