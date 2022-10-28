
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTMtime;
class DTMtimeRcd;
class DTRecoConditions;
class DTRecoConditionsVdriftRcd;
class DTMtimeValidateDBRead : public edm::one::EDAnalyzer<> {
public:
  explicit DTMtimeValidateDBRead(edm::ParameterSet const& p);
  explicit DTMtimeValidateDBRead(int i);
  ~DTMtimeValidateDBRead() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
  bool readLegacyVDriftDB;  // which DB to use
  edm::ESGetToken<DTMtime, DTMtimeRcd> dtmtTimeToken_;
  edm::ESGetToken<DTRecoConditions, DTRecoConditionsVdriftRcd> dtrecoCondToken_;
};
