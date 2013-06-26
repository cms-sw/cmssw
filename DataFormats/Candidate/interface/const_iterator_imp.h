#ifndef Candidate_const_iterator_imp_h
#define Candidate_const_iterator_imp_h

/* \class reco::candidate::const_iterator_imp
 *
 * \author Luca Lista, INFN
 *
 */
namespace reco {
  namespace candidate {
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
  }
}

#endif
