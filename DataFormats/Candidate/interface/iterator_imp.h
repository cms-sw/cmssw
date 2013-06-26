#ifndef Candidate_iterator_imp_h
#define Candidate_iterator_imp_h

/* \class reco::candidate::iterator_imp
 *
 * \author Luca Lista, INFN
 *
 */
namespace reco {
  namespace candidate {
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
  }
}

#endif
