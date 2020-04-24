#ifndef ParallelAnalysis_TSelectorAnalyzer_h
#define ParallelAnalysis_TSelectorAnalyzer_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "TList.h"

template<typename Algo>
class TSelectorAnalyzer : public edm::EDAnalyzer {
public:
  TSelectorAnalyzer( const edm::ParameterSet & cfg ) :
    list_(), algo_( 0, list_ ) {
  }
  void analyze( const edm::Event & evt, const edm::EventSetup & ) override {
    algo_.process( evt );
  }  
  void endJob() override {
    algo_.postProcess( list_ );
    algo_.terminate( list_ );
  }
private:
  TList list_;
  Algo algo_; 
};

#endif
