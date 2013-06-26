
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTRangeT0ValidateDBRead : public edm::EDAnalyzer {

 public:

  explicit  DTRangeT0ValidateDBRead(edm::ParameterSet const& p);
  explicit  DTRangeT0ValidateDBRead(int i) ;
  virtual ~ DTRangeT0ValidateDBRead();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob();

 private:

  std::string dataFileName;
  std::string elogFileName;

};
