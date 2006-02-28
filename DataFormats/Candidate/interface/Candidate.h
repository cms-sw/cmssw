#ifndef Candidate_Candidate_h
#define Candidate_Candidate_h
// $Id: Candidate.h,v 1.17 2006/02/21 10:37:32 llista Exp $
#include "DataFormats/Candidate/interface/Particle.h"
#include <boost/static_assert.hpp>
#include <vector>

namespace reco {
  
  template<typename T>
  struct component {
    BOOST_STATIC_ASSERT(false);
  };

  class Candidate : public Particle {
  private:
    typedef std::vector<Candidate> CandVector;

  public:
    typedef CandVector::size_type size_type;
    struct const_iterator;
    struct iterator;

    Candidate() : Particle() { }
    explicit Candidate( const Particle & p ) : Particle( p ) { }
    explicit Candidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      Particle( q, p4, vtx ) { }
    virtual ~Candidate();
    virtual Candidate * clone() const = 0;
    virtual const_iterator begin() const = 0;
    virtual const_iterator end() const = 0;
    virtual iterator begin() = 0;
    virtual iterator end() = 0;
    virtual int numberOfDaughters() const = 0;
    virtual const Candidate & daughter( size_type i ) const = 0;
    virtual Candidate & daughter( size_type i ) = 0;

  public:
    struct const_iterator_imp {
      typedef ptrdiff_t difference_type;
      const_iterator_imp() { } 
      virtual ~const_iterator_imp() { }
      virtual const_iterator_imp * clone() const = 0;
      virtual void increase() = 0;
      virtual void decrease() = 0;
      virtual void increase( difference_type d ) = 0;
      virtual void decrease( difference_type d ) = 0;
      virtual bool equal_to( const const_iterator_imp * ) const = 0;
      virtual bool less_than( const const_iterator_imp * ) const = 0;
      virtual void assign( const const_iterator_imp * ) = 0;
      virtual const Candidate & deref() const = 0;
      virtual difference_type difference( const const_iterator_imp * ) const = 0;
    };
    struct iterator_imp {
      typedef ptrdiff_t difference_type;
      iterator_imp() { }
      virtual ~iterator_imp() { }
      virtual iterator_imp * clone() const = 0;
      virtual const_iterator_imp * const_clone() const = 0;
      virtual void increase() = 0;
      virtual void decrease() = 0;
      virtual void increase( difference_type d ) = 0;
      virtual void decrease( difference_type d ) = 0;
      virtual bool equal_to( const iterator_imp * ) const = 0;
      virtual bool less_than( const iterator_imp * ) const = 0;
      virtual void assign( const iterator_imp * ) = 0;
      virtual Candidate & deref() const = 0;
      virtual difference_type difference( const iterator_imp * ) const = 0;
    };
    
  public:
    struct iterator;
    struct const_iterator {
      typedef Candidate value_type;
      typedef Candidate * pointer;
      typedef Candidate & reference;
      typedef ptrdiff_t difference_type;
      typedef CandVector::const_iterator::iterator_category iterator_category;
      const_iterator() { }
      const_iterator( const_iterator_imp * it ) : i( it ) { }
      const_iterator( const const_iterator & it ) : i( it.i->clone() ) { }
      const_iterator( const iterator & it ) : i( it.i->const_clone() ) { }
      ~const_iterator() { delete i; }
      const_iterator & operator=( const const_iterator & it ) { i->assign( it.i ); return *this; }
      const_iterator& operator++() { i->increase(); return *this; }
      const_iterator operator++( int ) { const_iterator ci = *this; i->increase(); return ci; }
      const_iterator& operator--() { i->decrease(); return *this; }
      const_iterator operator--( int ) { const_iterator ci = *this; i->decrease(); return ci; }
      difference_type operator-( const const_iterator & o ) const { return i->difference( o.i ); }
      const_iterator operator+( difference_type n ) const { 
	const_iterator_imp * ii = i->clone(); ii->increase( n );
	return const_iterator( ii ); 
      }
      const_iterator operator-( difference_type n ) const { 
	const_iterator_imp * ii = i->clone(); ii->decrease( n );
	return const_iterator( ii ); 
      }
      bool operator<( const const_iterator & o ) const { return i->less_than( o.i ); }
      bool operator==( const const_iterator& ci ) const { return i->equal_to( ci.i ); }
      bool operator!=( const const_iterator& ci ) const { return ! i->equal_to( ci.i ); }
      const Candidate & operator * () const { return i->deref(); }
      const Candidate * operator->() const { return & ( operator*() ); }
      const_iterator & operator +=( difference_type d ) { i->increase( d ); return *this; }
      const_iterator & operator -=( difference_type d ) { i->decrease( d ); return *this; }
    private:
      const_iterator_imp * i;
    };

    struct iterator {
      typedef Candidate value_type;
      typedef Candidate * pointer;
      typedef Candidate & reference;
      typedef ptrdiff_t difference_type;
      typedef CandVector::iterator::iterator_category iterator_category;
      iterator() { }
      iterator( iterator_imp * it ) : i( it ) { }
      iterator( const iterator & it ) : i( it.i->clone() ) { }
      ~iterator() { delete i; }
      iterator & operator=( const iterator & it ) { i->assign( it.i ); return *this; }
      iterator& operator++() { i->increase(); return *this; }
      iterator operator++( int ) { iterator ci = *this; i->increase(); return ci; }
      iterator& operator--() { i->increase(); return *this; }
      iterator operator--( int ) { iterator ci = *this; i->decrease(); return ci; }
      difference_type operator-( const iterator & o ) const { return i->difference( o.i ); }
      iterator operator+( difference_type n ) const { 
	iterator_imp * ii = i->clone(); ii->increase( n );
	return iterator( ii ); 
      }
      iterator operator-( difference_type n ) const { 
	iterator_imp * ii = i->clone(); ii->decrease( n );
	return iterator( ii ); 
      }
      bool operator<( const iterator & o ) { return i->less_than( o.i ) ; }
      bool operator==( const iterator& ci ) const { return i->equal_to( ci.i ); }
      bool operator!=( const iterator& ci ) const { return ! i->equal_to( ci.i ); }
      Candidate & operator * () const { return i->deref(); }
      Candidate * operator->() const { return & ( operator*() ); }
      iterator & operator +=( difference_type d ) { i->increase( d ); return *this; }
      iterator & operator -=( difference_type d ) { i->decrease( d ); return *this; }
    private:
      iterator_imp * i;
      friend const_iterator::const_iterator( const iterator & );
    };

    class setup {
    private:
      struct setupFlag {
	setupFlag( bool f = true ) : value( f ) { }
	bool value;
      };
      
    public:
      struct setupCharge : public setupFlag {
	setupCharge( bool f = true ) : setupFlag( f ) { }
      };
      
      struct setupP4 : public setupFlag {
	setupP4( bool f = true ) : setupFlag( f ) { }
      };
      
    public:
      setup( setupCharge q, setupP4 p ) : 
	modifyP4( p.value ), modifyCharge( q.value ) { }
      virtual ~setup();
      virtual void set( Candidate & c ) = 0;
      void setP4( LorentzVector & p ) const;
      void setCharge( Charge & q ) const;
      
    protected:
      LorentzVector p4;
      Charge charge;
      bool modifyP4, modifyCharge;
    };

    void set( setup & s );

  private:
    template<typename T> friend struct component; 
    friend class OverlapChecker;
    virtual bool overlap( const Candidate & ) const = 0;
  };

  inline void Candidate::set( setup & s ) {
    s.set( * this );
    s.setP4( p4_ );
    s.setCharge( q_ );
    // .. vertex or other attributes should be updated 
    // when will be implemented
  }
  
}

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#endif
