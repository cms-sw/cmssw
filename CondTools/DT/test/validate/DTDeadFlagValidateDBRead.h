
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTDeadFlagValidateDBRead : public edm::EDAnalyzer {
public:
  explicit DTDeadFlagValidateDBRead(edm::ParameterSet const& p);
  explicit DTDeadFlagValidateDBRead(int i);
  virtual ~DTDeadFlagValidateDBRead();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob();

private:
  std::string dataFileName;
  std::string elogFileName;
};
