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
 * \version $Id: CompositeRefCandidateT.h,v 1.15 2012/04/07 05:54:55 davidlt Exp $
 *
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

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
    virtual ~CompositeRefCandidateT();
    /// returns a clone of the candidate
    virtual CompositeRefCandidateT<D> * clone() const;
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter iterator
    virtual iterator end();
    /// number of daughters
    virtual size_t numberOfDaughters() const;
    /// number of mothers
    virtual size_t numberOfMothers() const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter(size_type) const;
    using ::reco::LeafCandidate::daughter; // avoid hiding the base
    /// return mother at a given position, i = 0, ... numberOfMothers() - 1 (read only mode)
    virtual const Candidate * mother(size_type = 0) const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter(size_type);
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


  private:
    /// const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    /// iterator implementation
    typedef candidate::iterator_imp_specific_dummy<daughters> iterator_imp_specific;
    /// collection of references to daughters
    daughters dau;
    /// collection of references to mothers
    daughters mom;
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
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
  Candidate::const_iterator CompositeRefCandidateT<D>::begin() const { 
    return const_iterator( new const_iterator_imp_specific( dau.begin() ) ); 
  }
  
  template<typename D>
  Candidate::const_iterator CompositeRefCandidateT<D>::end() const { 
    return const_iterator( new const_iterator_imp_specific( dau.end() ) ); 
  }    
  
  template<typename D>
  Candidate::iterator CompositeRefCandidateT<D>::begin() { 
    return iterator( new iterator_imp_specific ); 
  }
  
  template<typename D>
  Candidate::iterator CompositeRefCandidateT<D>::end() { 
    return iterator( new iterator_imp_specific ); 
  }    
  
  template<typename D>
  const Candidate * CompositeRefCandidateT<D>::daughter( size_type i ) const { 
    return ( i < numberOfDaughters() ) ? & * dau[ i ] : 0;
  }
  
  template<typename D>
  const Candidate * CompositeRefCandidateT<D>::mother( size_type i ) const { 
    return ( i < numberOfMothers() ) ? & * mom[ i ] : 0;
  }
  
  template<typename D>
  Candidate * CompositeRefCandidateT<D>::daughter( size_type i ) { 
    return 0;
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
