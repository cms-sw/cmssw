#ifndef RecoCandidate_RecoEcalCandidate_h
#define RecoCandidate_RecoEcalCandidate_h
/** \class reco::RecoEcalCandidate
 *
 * Reco Candidates with a Super Cluster component
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {

  class RecoEcalCandidate : public RecoCandidate {
  public:
    /// default constructor
    RecoEcalCandidate() : RecoCandidate() { }
    /// constructor from values
    RecoEcalCandidate( Charge q , const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		       int pdgId = 0, int status = 0 ) :
      RecoCandidate( q, p4, vtx, pdgId, status ) { }
    /// constructor from values
    RecoEcalCandidate( Charge q , const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		       int pdgId = 0, int status = 0 ) :
      RecoCandidate( q, p4, vtx, pdgId, status ) { }
    /// destructor
    ~RecoEcalCandidate() override;
    /// returns a clone of the candidate
    RecoEcalCandidate * clone() const override;
    /// set reference to superCluster
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// reference to a superCluster
    reco::SuperClusterRef superCluster() const override;

  private:
    /// check overlap with another candidate
    bool overlap( const Candidate & ) const override;
    /// reference to a superCluster
    reco::SuperClusterRef superCluster_;
  };
  
}

#endif
