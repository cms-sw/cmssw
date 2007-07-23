#ifndef TrackTSelector_h
#define TrackTSelector_h
/** \class TrackTSelector
 *
 * Simple interactive analysis example based on FWLite TSelector
 * accessing EDM data
 *
 * \author Luca Lista, INFN
 *
 * $Id: TrackTSelector.h,v 1.1 2006/06/27 09:53:33 llista Exp $
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include <TFile.h>
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelector.h"
#include "PhysicsTools/ParallelAnalysis/interface/TrackAnalysisAlgorithm.h"

class TCanvas;

namespace examples {

  class TrackTSelector : 
    public TFWLiteSelector<TrackAnalysisAlgorithm> {
  public :
    /// default constructor
    TrackTSelector();
    /// begin processing
    void begin( TList * & );
    /// terminate processing
    void terminate( TList & );
    
  private:
    /// avoid copy constructor
    TrackTSelector( const TrackTSelector & );
    /// avoid assignment operator
    TrackTSelector operator=( TrackTSelector const & );
    /// draw an histogram
    void draw( const TList &, const char * );
    /// canvas used as temporaty drawing object
    TCanvas * canvas_;
  };
  
}

#endif
