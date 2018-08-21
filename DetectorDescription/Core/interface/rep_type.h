#ifndef DETECTOR_DESCRIPTION_CORE_DDI_REP_TYPE_H
#define DETECTOR_DESCRIPTION_CORE_DDI_REP_TYPE_H

namespace DDI {
  
  template< class N, class I > 
    struct rep_traits
    {
      using name_type = N;
      using value_type = typename I::value_type;
      using pointer = typename I::pointer;
      using reference = typename I::reference;
    };
  
  template <class N, class I> 
    struct rep_traits< N, I* >
    {
      using name_type = N;
      using vlue_type = I;
      using pointer = I*;
      using reference = I&;
    };
  
  template< class N, class I >
    struct rep_type
    {
      rep_type() : second( nullptr ), init_(false) {}
      rep_type( const N & n, I i ) : first( n ), second( i ), init_( false ) 
      { if( i ) init_ = true; }
      N first;
      I second;
      
      const typename rep_traits< N, I >::name_type & name() const { return first; }
      const typename rep_traits< N, I >::reference rep()  const { return *second; }

      I swap( I i )
      { 
	I tmp( second );
	second = i;
	init_ = false;
	if( i ) init_ = true;
	return tmp;
      }
      operator bool() const { return init_; }
      
    private:
      bool init_;
   };
}

#endif
