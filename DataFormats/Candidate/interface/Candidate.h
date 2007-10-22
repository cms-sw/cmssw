#ifndef Candidate_Candidate_h
#define Candidate_Candidate_h
/** \class reco::Candidate
 *
 * generic reconstructed particle candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Candidate.h,v 1.35 2007/10/17 08:01:38 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/component.h"
#include "DataFormats/Candidate/interface/const_iterator.h"
#include "DataFormats/Candidate/interface/iterator.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "boost/iterator/filter_iterator.hpp"

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
    /// constructor from values
    Candidate( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
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
    virtual size_t numberOfMothers() const = 0;
    /// return pointer to mother
    virtual const Candidate * mother( size_t i = 0 ) const = 0;
    /// returns true if this candidate has a reference to a master clone.
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate
    virtual bool hasMasterClone() const;
    /// returns reference to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate
    virtual const CandidateBaseRef & masterClone() const;
    /// cast master clone reference to a concrete type
    template<typename Ref>
    Ref masterRef() const { return masterClone().template castTo<Ref>(); }
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

    template<typename S> 
    struct daughter_iterator {
      typedef boost::filter_iterator<S, const_iterator> type;
    };

    template<typename S>
    typename daughter_iterator<S>::type beginFilter( const S & s ) const {
      return boost::make_filter_iterator(s, begin(), end());
    }
    template<typename S>
    typename daughter_iterator<S>::type endFilter( const S & s ) const {
      return boost::make_filter_iterator(s, end(), end());
    }
  private:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & ) const = 0;
    template<typename, typename> friend struct component; 
    friend class OverlapChecker;
    friend class ShallowCloneCandidate;
  };

}

#endif
