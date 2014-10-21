#ifndef GEOMETRY_CALOGEOMETRY_EZArrayFL_H
#define GEOMETRY_CALOGEOMETRY_EZArrayFL_H 1

#include "Geometry/CaloGeometry/interface/EZMgrFL.h"

/** \class EZArrayFL<T>

  stl-vector-LIKE Class designed to allow many small fixed-length
  containers to have a common memory managed by a single higher-level object.

  It has the usual common iterators (begin, end) and functions (size, capacity, etc)
  but is NOT actually an STL-vector.

  It practices 'on-demand', or 'lazy evaluation', only allocating
  memory when requested.

*/


template < class T >
class EZArrayFL
{
   public:

      typedef          EZMgrFL< T >             MgrType ;
      typedef typename MgrType::iterator        iterator ;
      typedef typename MgrType::const_iterator  const_iterator ;
      typedef typename MgrType::reference       reference ;
      typedef typename MgrType::const_reference const_reference ;
      typedef typename MgrType::size_type       size_type ;
      typedef typename MgrType::value_type      value_type ;

      EZArrayFL< T >() : m_begin ( 0 ) ,
			 m_mgr   (   ) {}

      EZArrayFL< T >( MgrType* mgr  ) : m_begin ( 0 ) ,
					      m_mgr   ( mgr )   {}

      EZArrayFL< T >( MgrType* mgr   , 
		      const_iterator start ,
		      const_iterator finis       ) :
	 m_begin ( 0==finis-start ? (iterator)0 : mgr->assign() ) ,
	 m_mgr   ( mgr )
      {
	 assert( ( finis - start ) == m_mgr->subSize() ) ;
	 iterator i ( begin() ) ;
	 for( const_iterator ic ( start ) ; ic != finis ; ++ic )
	 {
	    (*i) = (*ic) ;
	 }
      }

      virtual ~EZArrayFL< T >() {}

      void resize() { assign() ; }

      void assign( const T& t = T() ) 
      {
	 assert( (iterator)0 == m_begin ) ;
	 m_begin = m_mgr->assign( t ) ;
      }

      const_iterator begin() const { return m_begin ; } 
      const_iterator end()   const { return m_begin + m_mgr->subSize() ; }

      reference operator[]( const unsigned int i ) 
      {
	 if( (iterator)0 == m_begin ) assign() ;
	 return *( m_begin + i ) ; 
      }

      const_reference operator[]( const unsigned int i ) const 
      {
	 return *( m_begin + i ) ;
      }

      bool uninitialized() const { return ( (iterator)0 == m_begin ) ;  }

      bool empty()         const { return ( 0 == size() ) ;  }

      size_type size()     const { return m_mgr->subSize() ; }

      size_type capacity() const { return size() ; }

   protected:

   private:

      //EZArrayFL( const EZArrayFL& ) ; //stop
      //EZArrayFL& operator=( const EZArrayFL& ) ; //stop
      iterator m_begin   ;
      MgrType*   m_mgr   ;
};

#endif
