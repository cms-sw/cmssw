#ifndef Candidate_CandidateWithRef_h
#define Candidate_CandidateWithRef_h
/** \class reco::CandidateWithRef
 *
 * Reco Candidates with a generic reference as component
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace reco {

  template<typename Ref>
  class CandidateWithRef : public LeafCandidate {
  public:
    typedef Ref reference;
    /// default constructor
    CandidateWithRef() : LeafCandidate() { }
    /// constructor from values
    CandidateWithRef( const LorentzVector & p4, Charge q = 0, const Point & vtx = Point( 0, 0, 0 ) ) :
      LeafCandidate( q, p4, vtx ) { }
    /// destructor
    virtual ~CandidateWithRef();
    /// returns a clone of the candidate
    virtual CandidateWithRef * clone() const;
    /// set reference
    void setRef( const Ref & r ) { ref_ = r; }
    /// reference 
    reference ref() const { return ref_; }

    CMS_CLASS_VERSION(13)

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a CaloRecHit
    reference ref_;
  };

  // the following has to be added for any single Ref type
  // GET_DEFAULT_CANDIDATE_COMPONENT( CandidateWithRef<Ref>, CandidateWithRef<Ref>::reference, ref )

  template<typename Ref>
    CandidateWithRef<Ref>::~CandidateWithRef() { 
  }
  
  template<typename Ref>
    CandidateWithRef<Ref> * CandidateWithRef<Ref>::clone() const {
    return new CandidateWithRef<Ref>( * this );
  }
  
  template<typename Ref>
    bool CandidateWithRef<Ref>::overlap( const Candidate & c ) const {
    const CandidateWithRef * o = dynamic_cast<const CandidateWithRef *>( & c );
    if ( o == 0 ) return false;
    if ( ref().isNull() ) return false;
    if ( o->ref().isNull() ) return false;
    return ( ref() != o->ref() );
  }
  
}

#endif
