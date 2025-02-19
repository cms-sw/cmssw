// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

//
// class declaration
//

#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

class CSCCableReadTest : public edm::EDAnalyzer {
 public:
  explicit CSCCableReadTest(const edm::ParameterSet&);
  ~CSCCableReadTest();
  
 private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
};
