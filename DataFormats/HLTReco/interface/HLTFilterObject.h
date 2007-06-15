#ifndef HLTReco_HLTFilterObject_h
#define HLTReco_HLTFilterObject_h

/** \class HLTFilterObject
 *
 *
 *  If HLT cuts of intermediate or final HLT filters are satisfied,
 *  instances of this class hold the combination of reconstructed
 *  physics objects (e/gamma/mu/jet/MMet...) satisfying the cuts.
 *
 *  This implementation is not completely space-efficient as some
 *  physics object containers may stay empty. However, the big
 *  advantage is that the solution is generic, i.e., works for all
 *  possible HLT filters. Hence we accept the reasonably small
 *  overhead of empty containers.
 *
 *  $Date: 2007/05/16 15:38:43 $
 *  $Revision: 1.25 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <algorithm>
#include <cassert>
#include <vector>
#include <map>

namespace reco
{
  class HLTFilterObjectBase {

  private:
    unsigned short int path_;    // index of path in trigger table
    unsigned short int module_;  // index of module on trigger path

  public:

    HLTFilterObjectBase(int path=0, int module=0) {
      assert( (0<=path  ) && (path  < 65536) );
      assert( (0<=module) && (module< 65536) );
      path_   = path;
      module_ = module;
    }

    unsigned short int path()   const { return path_  ;}
    unsigned short int module() const { return module_;}

  };

  class HLTFilterObject : public HLTFilterObjectBase {

  private:
    std::vector<Particle> particles_; // particles/MET (4-momentum vectors) used by filter

  public:

    HLTFilterObject(int path=0, int module=0)
      : HLTFilterObjectBase(path,module), particles_() { }

    unsigned int size() const { return particles_.size();}

    const Particle& getParticle(const unsigned int i) const {
      return particles_.at(i);
    }

    void putParticle(const Particle& ref) {
      particles_.push_back(ref);
    }

    void putParticle(const edm::RefToBase<Candidate>& ref) {
      particles_.push_back(Particle(*ref));
    }


    // Set methods for HLT Particle components (default = most-recent entry)

    void setCharge(const Particle::Charge& q, int i=-1) {
      if (i<0) {i+=size();}
      particles_.at(i).setCharge(q);
    }
    void setP4(const Particle::LorentzVector& p4, int i=-1) {
      if (i<0) {i+=size();}
      particles_.at(i).setP4(p4);
    }
    void setVertex(const Particle::Point& v3, int i=-1) {
      if (i<0) {i+=size();}
      particles_.at(i).setVertex(v3);
    }
    void setPdgId(const int& pdgId, int i=-1) {
      if (i<0) {i+=size();}
      particles_.at(i).setPdgId(pdgId);
    }
    void setStatus(const int& status, int i=-1) {
      if (i<0) {i+=size();}
      particles_.at(i).setStatus(status);
    }

  };


  class HLTFilterObjectWithRefs : public HLTFilterObject {

  private:
    std::vector<edm::RefToBase<Candidate> > refs_; // Refs into original collections
    std::vector<edm::ProductID>             pids_; // PIDs of AssociationMaps and alike

  public:

    HLTFilterObjectWithRefs(int path=0, int module=0)
      : HLTFilterObject(path,module), refs_(), pids_() { }

    // template for any ConcreteCollection
    template<typename C>
    void putParticle(const edm::RefProd<C>& refprod, unsigned int i) {
      putParticle(edm::RefToBase<Candidate>(edm::Ref<C>(refprod,i)));
    }
    // template specialisation for an HLTFilterObjectWithRefs
    void putParticle(const edm::RefProd<HLTFilterObjectWithRefs>& refprod, unsigned int i) {
      putParticle(refprod->getParticleRef(i));
    }
    // method to do the actual work
    void putParticle(const edm::RefToBase<Candidate>& ref) {
      this->HLTFilterObject::putParticle(ref);
      refs_.push_back(ref);
    }

    const edm::RefToBase<Candidate>& getParticleRef(const unsigned int i) const {
      return refs_.at(i);
    }

    void putPID(const edm::ProductID& pid) {
      pids_.push_back(pid);
    }

    unsigned int nPID() const {return pids_.size();}

    const edm::ProductID& getPID(const unsigned int i) const {
      return pids_.at(i);
    }

    //
    // vector-like methods allowing to access an object of type
    // HLTFilterObjectWithRefs like a ConcreteCollection of Candidates
    //

    // user should check validity of Ref element before dereferencing it
    const Candidate & at        (unsigned int i) const { return *(refs_.at(i)); }
    const Candidate & operator[](unsigned int i) const { return *(refs_[i]); }

    //
    // const_iterator class allowing to access an object of type
    // HLTFilterObjectWithRefs like a ConcreteCollection of Candidates
    //
    // the code is adapted from class const_iterator of
    // DataFormats/Common/interface/OwnVector.h
    //

    class const_iterator {

    public:
      typedef std::vector<edm::RefToBase<Candidate> > base;
      typedef base::size_type size_type;
      typedef       Candidate   value_type;
      typedef       Candidate * pointer;
      typedef       Candidate & reference;
      typedef const Candidate & const_reference;
      typedef       ptrdiff_t   difference_type;
      typedef base::const_iterator::iterator_category iterator_category;

      const_iterator( const base::const_iterator & it ) : i( it ) { }
      const_iterator( const       const_iterator & it ) : i( it.i ) { }
      const_iterator() {}

      const_iterator & operator=( const const_iterator & it ) { i = it.i; return *this; }

      const_iterator & operator++() { ++i; return *this; }
      const_iterator   operator++( int ) { const_iterator ci = *this; ++i; return ci; }
      const_iterator & operator--() { --i; return *this; }
      const_iterator   operator--( int ) { const_iterator ci = *this; --i; return ci; }

      difference_type  operator-( const const_iterator & o ) const { return i - o.i; }
      const_iterator   operator+( difference_type n ) const { return const_iterator( i + n ); }
      const_iterator   operator-( difference_type n ) const { return const_iterator( i - n ); }

      bool operator< ( const const_iterator & o ) const { return i <  o.i; }
      bool operator<=( const const_iterator & o ) const { return i <= o.i; }
      bool operator> ( const const_iterator & o ) const { return i >  o.i; }
      bool operator>=( const const_iterator & o ) const { return i >= o.i; }
      bool operator==( const const_iterator& ci ) const { return i == ci.i; }
      bool operator!=( const const_iterator& ci ) const { return i != ci.i; }

      const Candidate * operator->() const { return i->get();}
      // user should check validity of Pointer before dereferencing it
      const Candidate & operator* () const { return * (operator->());}

      const_iterator & operator +=( difference_type d ) { i += d; return *this; }
      const_iterator & operator -=( difference_type d ) { i -= d; return *this; }

    private:
      base::const_iterator i;
    };

    const_iterator begin() const { return const_iterator(refs_.begin());}
    const_iterator end()   const { return const_iterator(refs_.end()  );}
 
  };

}

#endif
