#ifndef RecoCandidate_CaloRecHitCandidate_h
#define RecoCandidate_CaloRecHitCandidate_h
/** \class reco::CaloRecHitCandidate
 *
 * Reco Candidates with a CaloRecHit component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CaloRecHitCandidate.h,v 1.2 2006/12/11 10:12:03 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace reco {

  class CaloRecHitCandidate : public LeafCandidate {
  public:
    typedef edm::RefToBase<CaloRecHit> CaloRecHitRef;
    /// default constructor
    CaloRecHitCandidate() : LeafCandidate() { }
    /// constructor from values
    CaloRecHitCandidate( const LorentzVector & p4, Charge q = 0, const Point & vtx = Point( 0, 0, 0 ) ) :
      LeafCandidate( q, p4, vtx ) { }
    /// constructor from values
    CaloRecHitCandidate( const PolarLorentzVector & p4, Charge q = 0, const Point & vtx = Point( 0, 0, 0 ) ) :
      LeafCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~CaloRecHitCandidate();
    /// returns a clone of the candidate
    virtual CaloRecHitCandidate * clone() const;
    /// set CaloRecHit reference
    void setCaloRecHit( const CaloRecHitRef & r ) { caloRecHit_ = r; }
    /// reference to a CaloRecHit
    CaloRecHitRef caloRecHit() const { return caloRecHit_; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a CaloRecHit
    CaloRecHitRef caloRecHit_;
  };
    /// get default Track component
  GET_DEFAULT_CANDIDATE_COMPONENT( CaloRecHitCandidate, edm::RefToBase<CaloRecHit>, caloRecHit );

}

#endif
