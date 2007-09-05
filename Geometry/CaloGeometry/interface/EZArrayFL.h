#ifndef GEOMETRY_CALOGEOMETRY_EZArrayFL_H
#define GEOMETRY_CALOGEOMETRY_EZArrayFL_H 1

#include "Geometry/CaloGeometry/interface/EZMgrFL.h"

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

      EZArrayFL< T >( const MgrType* mgr       , 
		      const T&       t    = T()  ) :
	 m_begin ( 0==size ? (iterator)0 : mgr->assign( t ) ) ,
	 m_mgr   ( mgr )   {}

      EZArrayFL< T >( const MgrType* mgr   , 
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

      virtual ~EZArrayFL< T >() { m_mgr->release( m_begin ) ; }

      virtual void resize() { m_mgr->assign() ; }

      virtual void assign( const T&  t = T() ) const 
      {
	 assert( (iterator)0 == m_begin ) ;
	 m_begin = m_mgr->assign( t ) ;
	 m_end   = m_begin + m_mgr->subSize() ;
      }

      const_iterator begin() const { return m_begin ; } 
      const_iterator end()   const { return m_begin + m_mgr->subSize() ; }

      reference operator[]( const unsigned int i ) 
      {
	 return *( m_begin + i ) ; 
      }

      const_reference operator[]( const unsigned int i ) const 
      {
	 return (reference)(*this)[i] ;
      }

      bool uninitialized() const { return ( 0 == m_begin ) ;  }

      bool empty()         const { return ( 0 == size() ) ;  }

      size_type size()     const { return m_mgr->subSize() ; }

      size_type capacity() const { return size() ; }

   protected:

   private:

      EZArrayFL< T >() ; //stop
      //EZArrayFL( const EZArrayFL& ) ; //stop
      //EZArrayFL& operator=( const EZArrayFL& ) ; //stop
      mutable iterator m_begin   ;
      const MgrType*   m_mgr   ;
};

#endif
