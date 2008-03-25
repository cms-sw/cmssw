
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTMtimeValidateDBRead : public edm::EDAnalyzer {

 public:

  explicit  DTMtimeValidateDBRead(edm::ParameterSet const& p);
  explicit  DTMtimeValidateDBRead(int i) ;
  virtual ~ DTMtimeValidateDBRead();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void endJob();

 private:

  std::string dataFileName;
  std::string elogFileName;

};
