#ifndef JetExamples_DummyCaloTowerAnalyzer_h
#define JetExamples_DummyCaloTowerAnalyzer_h
// $Id: DummyCaloTowerAnalyzer.h,v 1.1 2006/02/23 10:35:35 llista Exp $
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

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
