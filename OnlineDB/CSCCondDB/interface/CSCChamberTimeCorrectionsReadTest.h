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

class CSCChamberTimeCorrectionsReadTest : public edm::EDAnalyzer {
public:
  explicit CSCChamberTimeCorrectionsReadTest(const edm::ParameterSet&);
  ~CSCChamberTimeCorrectionsReadTest() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
};
