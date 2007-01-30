#ifndef ParallelAnalysis_TrackTSelectorAnalyzer_h
#define ParallelAnalysis_TrackTSelectorAnalyzer_h
/* class examples::TrackTSelectorAnalyzer
 *
 * Sample module that reuses algorithm class developed for a TSelector
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision$
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "PhysicsTools/ParallelAnalysis/interface/TrackAnalysisAlgorithm.h"
#include <TList.h>

namespace examples {

  class TrackTSelectorAnalyzer : public edm::EDAnalyzer {
  public:
    /// constructor
    TrackTSelectorAnalyzer( const edm::ParameterSet & );
  private:
    /// analyze an event
    void analyze( const edm::Event &, const edm::EventSetup & );
    /// end job
    void endJob();
    /// histogram list
    TList histograms_;
    /// algorithm 
    TrackAnalysisAlgorithm algo_;
  };

}

#endif
