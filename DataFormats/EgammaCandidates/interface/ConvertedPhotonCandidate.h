#ifndef EgammaCandidates_ConvertedPhotonCandidate_h
#define EgammaCandidates_ConvertedPhotonCandidate_h
/** \class reco::ConvertedPhotonCandidate ConvertedPhotonCandidate.h DataFormats/EgammaCandidates/interface/ConvertedPhotonCandidate.h
 *
 * Reco Candidates with an ConvertedPhoton component
 *
 * \author N.Marinelli  University of Notre Dame, US
 *
 * \version $Id: ConvertedPhotonCandidate.h,v 1.1 2006/06/09 14:19:30 nancy Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ConvertedPhotonCandidateFwd.h"

namespace reco {

  class ConvertedPhotonCandidate : public RecoCandidate {
  public:
    /// default constructor
    ConvertedPhotonCandidate() : RecoCandidate() { }
    /// constructor from values
    ConvertedPhotonCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~ConvertedPhotonCandidate();
    /// returns a clone of the candidate
    virtual ConvertedPhotonCandidate * clone() const;
    /// reference to a SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// set refrence to ConvertedPhoton component
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
  };
  
}

#endif
