
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTT0ValidateDBRead : public edm::EDAnalyzer {
public:
  explicit DTT0ValidateDBRead(edm::ParameterSet const& p);
  explicit DTT0ValidateDBRead(int i);
  ~DTT0ValidateDBRead() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
};
