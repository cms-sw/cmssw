#ifndef ParallelAnalysis_TrackAnalysisAlgorithm_h
#define ParallelAnalysis_TrackAnalysisAlgorithm_h
/* class examples::TrackAnalysisAlgorithm
 *
 * Algorithm structure siutable for both TSelector running
 * and Framework processing
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision$
 */

class TH1F;
class TList;
namespace edm { class Event; }

namespace examples {

  struct TrackAnalysisAlgorithm {
    /// constructor
    TrackAnalysisAlgorithm( const TList *, TList& );
    /// process one event
    void process( const edm::Event&  );
    /// post process
    void postProcess( TList & );
    /// histograms
    TH1F * h_pt, * h_eta;
    /// histogram names
    static const char * kPt, * kEta;
  };

}

#endif
