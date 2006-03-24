#ifndef MuonReco_MuonExtra_h
#define MuonReco_MuonExtra_h
//
// $Id: MuonExtra.h,v 1.1 2006/02/20 14:41:58 llista Exp $
//
// Definition of TrackExtra class for RECO
//
// Author: Luca Lista
//
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/MuonReco/interface/MuonExtraFwd.h"
#include"DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class MuonExtra : public TrackExtra {
  public:
    MuonExtra() { }
    MuonExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok );

    void setStandAloneMuon( const TrackRef & ref ) { standAloneMuon_ = ref; }
    const TrackRef & standAloneMuon() const { return standAloneMuon_; }

  private:
    TrackRef standAloneMuon_;
  };

}

#endif
