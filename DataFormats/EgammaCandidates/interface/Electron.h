#ifndef EgammaCandidates_Electron_h
#define EgammaCandidates_Electron_h
/** \class reco::Electron 
 *
 * Reco Candidates with an Electron component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Electron.h,v 1.10 2007/07/31 15:20:03 ratnik Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class Electron : public RecoCandidate {
  public:
    /// default constructor
    Electron() : RecoCandidate() { }
    /// constructor from values
    Electron( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx, -11 * q ) { }
    /// destructor
    virtual ~Electron();
    /// returns a clone of the candidate
    virtual Electron * clone() const;
    /// refrence to a Track
    virtual reco::TrackRef track() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// set refrence to Photon component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// set refrence to Track component
    void setTrack( const reco::TrackRef & r ) { track_ = r; }

    bool isElectron() const;
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
