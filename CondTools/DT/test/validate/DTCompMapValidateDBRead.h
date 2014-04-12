
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTCompMapValidateDBRead : public edm::EDAnalyzer {

 public:

  explicit  DTCompMapValidateDBRead(edm::ParameterSet const& p);
  explicit  DTCompMapValidateDBRead(int i) ;
  virtual ~ DTCompMapValidateDBRead();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob();

 private:

  std::string dataFileName;
  std::string elogFileName;

};

