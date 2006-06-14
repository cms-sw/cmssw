#include "FWCore/Framework/interface/EDAnalyzer.h"
namespace edm{
  class ParameterSet;
  class Event;
  class EventSetup;
}

//
// class decleration
//

class writeMultipleRecords : public edm::EDAnalyzer {
 public:
  explicit writeMultipleRecords(const edm::ParameterSet& iConfig );
  ~writeMultipleRecords();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void endJob(){}
 private:
  // ----------member data ---------------------------
};

