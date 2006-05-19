#ifndef EgammaCandidates_ElectronCandidate_h
#define EgammaCandidates_ElectronCandidate_h
/** \class reco::ElectronCandidate ElectronCandidate.h DataFormats/EgammaCandidates/interface/ElectronCandidate.h
 *
 * Reco Candidates with an Electron component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ElectronCandidate.h,v 1.2 2006/04/26 07:56:19 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ElectronCandidateFwd.h"

namespace reco {

  class ElectronCandidate : public RecoCandidate {
  public:
    /// default constructor
    ElectronCandidate() : RecoCandidate() { }
    /// constructor from values
    ElectronCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~ElectronCandidate();
    /// returns a clone of the candidate
    virtual ElectronCandidate * clone() const;
    /// refrence to a Track
    virtual reco::TrackRef track() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// set refrence to Photon component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// set refrence to Track component
    void setTrack( const reco::TrackRef & r ) { track_ = r; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    /// reference to a Track
    reco::TrackRef track_;
  };
  
}

#endif
