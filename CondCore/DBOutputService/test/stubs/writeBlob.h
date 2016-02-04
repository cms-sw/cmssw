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

class writeBlob : public edm::EDAnalyzer {
 public:
  explicit writeBlob(const edm::ParameterSet& iConfig );
  ~writeBlob();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void endJob(){}
 private:
  std::string m_StripRecordName;
};

