
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTStatusFlagValidateDBRead : public edm::EDAnalyzer {

 public:

  explicit  DTStatusFlagValidateDBRead(edm::ParameterSet const& p);
  explicit  DTStatusFlagValidateDBRead(int i) ;
  virtual ~ DTStatusFlagValidateDBRead();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob();

 private:

  std::string dataFileName;
  std::string elogFileName;

};

