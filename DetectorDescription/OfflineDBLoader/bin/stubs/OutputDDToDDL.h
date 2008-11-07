#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <ostream>

class OutputDDToDDL : public edm::EDAnalyzer {

 public:
  explicit OutputDDToDDL( const edm::ParameterSet& iConfig );
  ~OutputDDToDDL();
  virtual void beginJob( edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};

 private:
  int rotNumSeed_;
  std::string fname_;
  std::ostream* xos_;
};
