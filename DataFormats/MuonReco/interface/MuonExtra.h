#ifndef MuonReco_MuonExtra_h
#define MuonReco_MuonExtra_h
/** \class reco::MuonExtra
 *  
 * Extension of a reconstructed Muon. It is ment to be stored
 * in the RECO, and to be referenced by its corresponding
 * object stored in the AOD
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 *
 */
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/MuonReco/interface/MuonExtraFwd.h"
#include"DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class MuonExtra : public TrackExtra {
  public:
    /// default constructor
    MuonExtra() { }
    /// constructor from outermost position and momentum
    MuonExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok );
    /// set reference to Track reconstructed in the muon detector only
    void setStandAloneMuon( const TrackRef & ref ) { standAloneMuon_ = ref; }
    /// reference to Track reconstructed in the muon detector only
    const TrackRef & standAloneMuon() const { return standAloneMuon_; }

  private:
    /// reference to Track reconstructed in the muon detector only
    TrackRef standAloneMuon_;
  };

}

#endif
