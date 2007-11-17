#ifndef RecoCandidate_RecoEcalCandidate_h
#define RecoCandidate_RecoEcalCandidate_h
/** \class reco::RecoEcalCandidate
 *
 * Reco Candidates with a Super Cluster component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: RecoEcalCandidate.h,v 1.4 2006/10/26 09:14:55 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoEcalCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoEcalCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoEcalCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) :
      RecoCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~RecoEcalCandidate();
    /// returns a clone of the candidate
    virtual RecoEcalCandidate * clone() const;
    /// set reference to superCluster
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// reference to a superCluster
    virtual reco::SuperClusterRef superCluster() const;

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a superCluster
    reco::SuperClusterRef superCluster_;
  };
  
}

#endif
