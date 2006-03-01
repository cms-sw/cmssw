#ifndef RecoCandidate_RecoElectronCandidate_h
#define RecoCandidate_RecoElectronCandidate_h
/** \class reco::RecoElectronCandidate
 *
 * Reco Candidates with an Electron component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoElectronCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoElectronCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoElectronCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~RecoElectronCandidate();
    /// returns a clone of the candidate
    virtual RecoElectronCandidate * clone() const;
    /// set reference to Electron component
    void setElectron( const reco::ElectronRef & r ) { electron_ = r; }

  private:
    /// reference to an Electron
    virtual reco::ElectronRef electron() const;
    /// refrence to a Track
    virtual reco::TrackRef track() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// reference to an Electron
    reco::ElectronRef electron_;
  };
  
}

#endif
