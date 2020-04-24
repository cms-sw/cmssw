#ifndef Candidate_CompositeRefCandidateT_h
#define Candidate_CompositeRefCandidateT_h
/** \class reco::CompositeRefCandidateT<D>
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::Ref <...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

namespace reco {

  template<typename D>
  class CompositeRefCandidateT : public LeafCandidate {
  public:
    /// collection of references to daughters
    typedef D daughters;
    /// collection of references to daughters
    typedef D mothers;
    /// default constructor
    CompositeRefCandidateT() : LeafCandidate() { }
    /// constructor from values
    CompositeRefCandidateT( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			    int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from values
    CompositeRefCandidateT( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			    int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from a particle
    explicit CompositeRefCandidateT( const LeafCandidate& c ) : LeafCandidate( c ) { }
    /// destructor
    ~CompositeRefCandidateT() override;
    /// returns a clone of the candidate
    CompositeRefCandidateT<D> * clone() const override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// number of mothers
    size_t numberOfMothers() const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    const Candidate * daughter(size_type) const override;
    using ::reco::LeafCandidate::daughter; // avoid hiding the base
    /// return mother at a given position, i = 0, ... numberOfMothers() - 1 (read only mode)
    const Candidate * mother(size_type = 0) const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    Candidate * daughter(size_type) override;
    /// add a daughter via a reference
    void addDaughter( const typename daughters::value_type & );    
    /// add a daughter via a reference
    void addMother( const typename mothers::value_type & );    
    /// clear daughter references
    void clearDaughters() { dau.clear(); }
    /// clear mother references
    void clearMothers() { mom.clear(); }
    /// reference to daughter at given position       
    typename daughters::value_type daughterRef( size_type i ) const { return dau[ i ]; }
    /// references to daughtes
    const daughters & daughterRefVector() const { return dau; }
    /// reference to mother at given position
    typename daughters::value_type motherRef( size_type i = 0 ) const { return mom[ i ]; }
    /// references to mothers
    const mothers & motherRefVector() const { return mom; }
    /// set daughters product ID
    void resetDaughters( const edm::ProductID & id ) { dau = daughters( id ); }
    /// set mother product ID
    void resetMothers( const edm::ProductID & id ) { mom = mothers( id ); }

    CMS_CLASS_VERSION(13)

  private:
    /// collection of references to daughters
    daughters dau;
    /// collection of references to mothers
    daughters mom;
    /// check overlap with another candidate
    bool overlap( const Candidate & ) const override;
  };

  template<typename D>
  inline void CompositeRefCandidateT<D>::addDaughter( const typename daughters::value_type & cand ) { 
    dau.push_back( cand ); 
  }

  template<typename D>
  inline void CompositeRefCandidateT<D>::addMother( const typename daughters::value_type & cand ) { 
    mom.push_back( cand ); 
  }

  template<typename D>
  CompositeRefCandidateT<D>::~CompositeRefCandidateT() { 
  }
  
  template<typename D>
  CompositeRefCandidateT<D> * CompositeRefCandidateT<D>::clone() const { 
    return new CompositeRefCandidateT( * this ); 
  }
  
  template<typename D>
  const Candidate * CompositeRefCandidateT<D>::daughter( size_type i ) const { 
    return ( i < numberOfDaughters() ) ? & * dau[ i ] : nullptr;
  }
  
  template<typename D>
  const Candidate * CompositeRefCandidateT<D>::mother( size_type i ) const { 
    return ( i < numberOfMothers() ) ? & * mom[ i ] : nullptr;
  }
  
  template<typename D>
  Candidate * CompositeRefCandidateT<D>::daughter( size_type i ) { 
    return nullptr;
  }
  
  template<typename D>
  size_t CompositeRefCandidateT<D>::numberOfDaughters() const { 
    return dau.size(); 
  }
  
  template<typename D>
  size_t CompositeRefCandidateT<D>::numberOfMothers() const { 
    return mom.size(); 
  }
  
  template<typename D>
  bool CompositeRefCandidateT<D>::overlap( const Candidate & c2 ) const {
    throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeRefCanddate";
  }
}

#endif
