
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTRangeT0ValidateDBRead : public edm::EDAnalyzer {
public:
  explicit DTRangeT0ValidateDBRead(edm::ParameterSet const& p);
  explicit DTRangeT0ValidateDBRead(int i);
  ~DTRangeT0ValidateDBRead() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void endJob() override;

private:
  std::string dataFileName;
  std::string elogFileName;
};
