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
 * \version $Id: MuonExtra.h,v 1.3 2006/03/01 13:08:08 llista Exp $
 *
 */
#include "DataFormats/TrackReco/interface/TrackExtraBase.h"
#include "DataFormats/MuonReco/interface/MuonExtraFwd.h"
#include"DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class MuonExtra : public TrackExtraBase {
  public:
    /// default constructor
    MuonExtra() : TrackExtraBase() { }
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
