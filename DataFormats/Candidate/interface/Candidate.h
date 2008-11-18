#ifndef Candidate_Candidate_h
#define Candidate_Candidate_h
/** \class reco::Candidate
 *
 * generic reconstructed particle candidate
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Candidate.h,v 1.43 2008/04/22 10:59:31 cbern Exp $
 *
 */
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/component.h"
#include "DataFormats/Candidate/interface/const_iterator.h"
#include "DataFormats/Candidate/interface/iterator.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/Error.h"
#include "boost/iterator/filter_iterator.hpp"

class OverlapChecker;

namespace reco {
  
  class Candidate : public Particle {
  public:
    /// size type
    typedef size_t size_type;
    typedef candidate::const_iterator const_iterator;
    typedef candidate::iterator iterator;
    /// error matrix dimension
    enum { dimension = 3 };
    /// covariance error matrix (3x3)
    typedef math::Error<dimension>::type CovarianceMatrix;
    /// matix size
    enum { size = dimension * (dimension + 1)/2 };
    /// index type
    typedef unsigned int index;
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
    virtual size_type numberOfDaughters() const = 0;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter( size_type i ) const = 0;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter( size_type i ) = 0;
    /// number of mothers (zero or one in most of but not all the cases)
    virtual size_type numberOfMothers() const = 0;
    /// return pointer to mother
    virtual const Candidate * mother( size_type i = 0 ) const = 0;
    /// return the number of source Candidates 
    /// ( the candidates used to construct this Candidate)
    virtual size_type numberOfSourceCandidateRefs() const {return 0;} 
    /// return a RefToBase to one of the source Candidates 
    /// ( the candidates used to construct this Candidate)
    virtual CandidateBaseRef sourceCandidateRef( size_type i ) const {
      return CandidateBaseRef();
    }
    /// chi-squares
    virtual double vertexChi2() const;
    /** Number of degrees of freedom
     *  Meant to be Double32_t for soft-assignment fitters: 
     *  tracks may contribute to the vertex with fractional weights.
     *  The ndof is then = to the sum of the track weights.
     *  see e.g. CMS NOTE-2006/032, CMS NOTE-2004/002
     */
    virtual double vertexNdof() const;
    /// chi-squared divided by n.d.o.f.
    virtual double vertexNormalizedChi2() const;
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    virtual double vertexCovariance(int i, int j) const;
    /// return SMatrix
    CovarianceMatrix vertexCovariance() const { CovarianceMatrix m; fillVertexCovariance(m); return m; }
    /// fill SMatrix
    virtual void fillVertexCovariance(CovarianceMatrix & v) const;
    /// returns true if this candidate has a reference to a master clone.
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate
    virtual bool hasMasterClone() const;
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate
    virtual const CandidateBaseRef & masterClone() const;
    /// returns true if this candidate has a ptr to a master clone.
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate
    virtual bool hasMasterClonePtr() const;
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate
    virtual const CandidatePtr & masterClonePtr() const;
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
    template<typename T> T get( size_type i ) const { 
      if ( hasMasterClone() ) return masterClone()->get<T>( i );
      else return reco::get<T>( * this, i ); 
    }
    /// get a component
    template<typename T, typename Tag> T get( size_type i ) const { 
      if ( hasMasterClone() ) return masterClone()->get<T, Tag>( i );
      else return reco::get<T, Tag>( * this, i ); 
    }
    /// number of components
    template<typename T> size_type numberOf() const { 
      if ( hasMasterClone() ) return masterClone()->numberOf<T>();
      else return reco::numberOf<T>( * this ); 
    }
    /// number of components
    template<typename T, typename Tag> size_type numberOf() const { 
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

    virtual bool isElectron() const;
    virtual bool isMuon() const;
    virtual bool isStandAloneMuon() const;
    virtual bool isGlobalMuon() const;
    virtual bool isTrackerMuon() const;
    virtual bool isCaloMuon() const;
    virtual bool isPhoton() const;
    virtual bool isConvertedPhoton() const;
    virtual bool isJet() const;

  private:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & ) const = 0;
    template<typename, typename, typename> friend struct component; 
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;
  };

}

#endif
