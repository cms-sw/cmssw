
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTMtime;
class DTMtimeRcd;
class DTRecoConditions;
class DTRecoConditionsVdriftRcd;
class DTMtimeValidateDBRead : public edm::EDAnalyzer {
public:
  explicit DTMtimeValidateDBRead(edm::ParameterSet const& p);
  explicit DTMtimeValidateDBRead(int i);
  virtual ~DTMtimeValidateDBRead();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob();

private:
  std::string dataFileName;
  std::string elogFileName;
  bool readLegacyVDriftDB;  // which DB to use
  edm::ESGetToken<DTMtime, DTMtimeRcd> dtmtTimeToken_;
  edm::ESGetToken<DTRecoConditions, DTRecoConditionsVdriftRcd> dtrecoCondToken_;
};
