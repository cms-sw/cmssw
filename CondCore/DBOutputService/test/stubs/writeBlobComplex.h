#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
namespace edm{
  class ParameterSet;
  class Event;
  class EventSetup;
}

//
// class decleration
//

class writeBlobComplex : public edm::EDAnalyzer {
 public:
  explicit writeBlobComplex(const edm::ParameterSet& iConfig );
  ~writeBlobComplex();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void endJob(){}
 private:
  std::string m_RecordName;
};

