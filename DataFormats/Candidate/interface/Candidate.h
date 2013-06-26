#ifndef Candidate_Candidate_h
#define Candidate_Candidate_h
/** \class reco::Candidate
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), Benedikt Hegner (CERN)
 *
 *
 */
#include "DataFormats/Candidate/interface/component.h"
#include "DataFormats/Candidate/interface/const_iterator.h"
#include "DataFormats/Candidate/interface/iterator.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/Error.h"
#include "boost/iterator/filter_iterator.hpp"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Rtypes.h"

#include "DataFormats/Candidate/interface/Particle.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


class OverlapChecker;

namespace reco {
  
  class Candidate {
  public:
    typedef size_t size_type;
    typedef candidate::const_iterator const_iterator;
    typedef candidate::iterator iterator;
 
    /// electric charge type
    typedef int Charge;
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// Lorentz vector
    typedef math::PtEtaPhiMLorentzVector PolarLorentzVector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// point in the space
    typedef math::XYZVector Vector;

    enum { dimension = 3 };
    /// covariance error matrix (3x3)
    typedef math::Error<dimension>::type CovarianceMatrix;
    /// matix size
    enum { size = dimension * (dimension + 1)/2 };
    /// index type
    typedef unsigned int index;

    /// default constructor
    Candidate() {};
    /// destructor
    virtual ~Candidate();
    /// electric charge
    virtual int charge() const = 0;
    /// set electric charge
    virtual void setCharge( Charge q ) = 0;
    /// electric charge
    virtual int threeCharge() const = 0;
    /// set electric charge
    virtual void setThreeCharge( Charge qx3 ) = 0;
    /// four-momentum Lorentz vector
    virtual const LorentzVector & p4() const = 0;
    /// four-momentum Lorentz vector
    virtual const PolarLorentzVector & polarP4() const = 0;
    /// spatial momentum vector
    virtual Vector momentum() const = 0;
    /// boost vector to boost a Lorentz vector 
    /// to the particle center of mass system
    virtual Vector boostToCM() const = 0;
    /// magnitude of momentum vector
    virtual double p() const = 0;
    /// energy
    virtual double energy() const = 0;
    /// transverse energy 
    virtual double et() const = 0;
    /// mass
    virtual float mass() const = 0;
    /// mass squared
    virtual float massSqr() const = 0;
    /// transverse mass
    virtual double mt() const = 0;
    /// transverse mass squared
    virtual double mtSqr() const = 0;
    /// x coordinate of momentum vector
    virtual double px() const = 0;
    /// y coordinate of momentum vector
    virtual double py() const = 0;
    /// z coordinate of momentum vector
    virtual double pz() const = 0;
    /// transverse momentum
    virtual float pt() const = 0;
    /// momentum azimuthal angle
    virtual float phi() const = 0;
    /// momentum polar angle
    virtual double theta() const = 0;
    /// momentum pseudorapidity
    virtual float eta() const = 0;
    /// rapidity
    virtual double rapidity() const = 0;
    /// rapidity
    virtual double y() const = 0;
    /// set 4-momentum
    virtual void setP4( const LorentzVector & p4 ) = 0;
    /// set 4-momentum
    virtual void setP4( const PolarLorentzVector & p4 )  = 0;
    /// set particle mass
    virtual void setMass( double m )  = 0;
    virtual void setPz( double pz )  = 0;
    /// vertex position
    virtual const Point & vertex() const  = 0;
    /// x coordinate of vertex position
    virtual double vx() const  = 0;
    /// y coordinate of vertex position
    virtual double vy() const  = 0;
    /// z coordinate of vertex position
    virtual double vz() const  = 0;
    /// set vertex
    virtual void setVertex( const Point & vertex )  = 0;
    /// PDG identifier
    virtual int pdgId() const  = 0;
    // set PDG identifier
    virtual void setPdgId( int pdgId )  = 0;
    /// status word
    virtual int status() const  = 0;
    /// set status word
    virtual void setStatus( int status )  = 0;
    /// set long lived flag
    virtual void setLongLived()  = 0;
    /// is long lived?
    virtual bool longLived() const  = 0;
    /// set mass constraint flag
    virtual void setMassConstraint() = 0;
    /// do mass constraint?
    virtual bool massConstraint() const = 0;
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
    /// return daughter with a specified role name
    virtual Candidate * daughter(const std::string& s ) = 0;
    /// return daughter with a specified role name
    virtual const Candidate * daughter(const std::string& s ) const = 0;
    /// number of mothers (zero or one in most of but not all the cases)
    virtual size_type numberOfMothers() const = 0;
    /// return pointer to mother
    virtual const Candidate * mother( size_type i = 0 ) const = 0;
    /// return the number of source Candidates 
    /// ( the candidates used to construct this Candidate)
    virtual size_t numberOfSourceCandidatePtrs() const  = 0;
    /// return a Ptr to one of the source Candidates 
    /// ( the candidates used to construct this Candidate)
    virtual CandidatePtr sourceCandidatePtr( size_type i ) const {
      return CandidatePtr();
    }
    /// \brief Set the ptr to the source Candidate. 
    /// 
    /// necessary, to allow a parallel treatment of all candidates 
    /// in PF2PAT. Does nothing for most Candidate classes, including 
    /// CompositePtrCandidates, where the source information is in fact
    /// the collection of ptrs to daughters. For non-Composite Candidates, 
    /// this function can be used to set the ptr to the source of the 
    /// Candidate, which will allow to keep track 
    /// of the reconstruction history. 
    virtual void setSourceCandidatePtr( const CandidatePtr& ptr ) {};

    /// chi-squares
    virtual double vertexChi2() const = 0;
    /** Number of degrees of freedom
     *  Meant to be Double32_t for soft-assignment fitters: 
     *  tracks may contribute to the vertex with fractional weights.
     *  The ndof is then = to the sum of the track weights.
     *  see e.g. CMS NOTE-2006/032, CMS NOTE-2004/002
     */
    virtual double vertexNdof() const = 0;
    /// chi-squared divided by n.d.o.f.
    virtual double vertexNormalizedChi2() const = 0;
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    virtual double vertexCovariance(int i, int j) const = 0;
    /// fill SMatrix
    virtual CovarianceMatrix vertexCovariance() const { CovarianceMatrix m; fillVertexCovariance(m); return m; }  //TODO
    virtual void fillVertexCovariance(CovarianceMatrix & v) const = 0;
    /// returns true if this candidate has a reference to a master clone.
    /// This only happens if the concrete Candidate type is ShallowCloneCandidate
    virtual bool hasMasterClone() const = 0;
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowCloneCandidate
    virtual const CandidateBaseRef & masterClone() const = 0;
    /// returns true if this candidate has a ptr to a master clone.
    /// This only happens if the concrete Candidate type is ShallowClonePtrCandidate
    virtual bool hasMasterClonePtr() const = 0;
    /// returns ptr to master clone, if existing.
    /// Throws an exception unless the concrete Candidate type is ShallowClonePtrCandidate
    virtual const CandidatePtr & masterClonePtr() const = 0;
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

    virtual bool isElectron() const = 0;
    virtual bool isMuon() const = 0;
    virtual bool isStandAloneMuon() const = 0;
    virtual bool isGlobalMuon() const = 0;
    virtual bool isTrackerMuon() const = 0;
    virtual bool isCaloMuon() const = 0;
    virtual bool isPhoton() const = 0;
    virtual bool isConvertedPhoton() const = 0;
    virtual bool isJet() const = 0;

  protected:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & ) const = 0;
    template<typename, typename, typename> friend struct component; 
    friend class ::OverlapChecker;
    friend class ShallowCloneCandidate;
    friend class ShallowClonePtrCandidate;

  };

}

#endif
