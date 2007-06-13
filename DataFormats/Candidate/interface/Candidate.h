#ifndef Candidate_Candidate_h
#define Candidate_Candidate_h
/** \class reco::Candidate
 *
 * generic reconstructed particle candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Candidate.h,v 1.28 2007/06/12 21:27:21 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/component.h"
#include "DataFormats/Candidate/interface/const_iterator.h"
#include "DataFormats/Candidate/interface/iterator.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace reco {
  
  class Candidate : public Particle {
  public:
    /// size type
    typedef size_t size_type;
    typedef candidate::const_iterator const_iterator;
    typedef candidate::iterator iterator;

    /// default constructor
    Candidate() : Particle() { }
    /// constructor from a Particle
    explicit Candidate( const Particle & p ) : Particle( p ) { }
    /// constructor from values
    Candidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
	       int pdgId = 0, int status = 0, bool integerCharge = true ) : 
      Particle( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// destructor
    virtual ~Candidate();
    /// returns a clone of the Candidate object
    virtual Candidate * clone() const = 0;
    /// first daughter const_iterator
    virtual const_iterator begin() const = 0;
    /// last daughter const_iterator
    virtual const_iterator end() const = 0;
    /// first daughter iterator
    virtual iterator begin() = 0;
    /// last daughter iterator
    virtual iterator end() = 0;
    /// number of daughters
    virtual size_t numberOfDaughters() const = 0;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter( size_type i ) const = 0;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter( size_type i ) = 0;
    /// number of mothers (zero or one in most of but not all the cases)
    unsigned int numberOfMothers() const { return mothers_.size(); }
    /// return pointer to mother
    const Candidate * mother( unsigned int i = 0 ) const { 
      return i < numberOfMothers() ? mothers_[ i ] : 0; 
    }
    /// returns true if this candidate has a reference to a master clone.
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate
    virtual bool hasMasterClone() const;
    /// returns reference to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate
    virtual const CandidateBaseRef & masterClone() const;
    /// get a component
    template<typename T> T get() const { 
      if ( hasMasterClone() ) return masterClone()->get<T>();
      else return reco::get<T>( * this ); 
    }
    /// get a component
    template<typename T, typename Tag> T get() const { 
      if ( hasMasterClone() ) return masterClone()->get<T, Tag>();
      else return reco::get<T, Tag>( * this ); 
    }
    /// get a component
    template<typename T> T get( size_t i ) const { 
      if ( hasMasterClone() ) return masterClone()->get<T>( i );
      else return reco::get<T>( * this, i ); 
    }
    /// get a component
    template<typename T, typename Tag> T get( size_t i ) const { 
      if ( hasMasterClone() ) return masterClone()->get<T, Tag>( i );
      else return reco::get<T, Tag>( * this, i ); 
    }
    /// number of components
    template<typename T> size_t numberOf() const { 
      if ( hasMasterClone() ) return masterClone()->numberOf<T>();
      else return reco::numberOf<T>( * this ); 
    }
    /// number of components
    template<typename T, typename Tag> size_t numberOf() const { 
      if ( hasMasterClone() ) return masterClone()->numberOf<T, Tag>();
      else return reco::numberOf<T, Tag>( * this ); 
    }

    /// add a new mother pointer
    void addMother( const Candidate * mother ) const {
      mothers_.push_back( mother );
    }

  private:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & ) const = 0;
    template<typename, typename> friend struct component; 
    friend class OverlapChecker;
    friend class ShallowCloneCandidate;
    /// mother link
    mutable std::vector<const Candidate *> mothers_;
    /// post-read fixup
    virtual void fixup() const = 0;
    /// declare friend class
    friend class edm::helpers::PostReadFixup;
  };

}

#endif
