#ifndef JetExamples_DummyCaloTowerAnalyzer_h
#define JetExamples_DummyCaloTowerAnalyzer_h
// $Id$
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "PhysicsTools/RecoCandidate/interface/RecoCaloTowerCandidate.h"

class DummyCaloTowerAnalyzer : public edm::EDAnalyzer {
 public:
  DummyCaloTowerAnalyzer( const edm::ParameterSet & ) { }
  ~DummyCaloTowerAnalyzer() { }
  void endJob() {
    reco::RecoCaloTowerCandidate r;
    if ( r.mass() > 0 ) return;
  }
private:
  void analyze( const edm::Event& , const edm::EventSetup& ) { }
};

#endif
