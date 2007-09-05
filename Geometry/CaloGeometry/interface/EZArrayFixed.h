#ifndef GEOMETRY_CALOGEOMETRY_EZArrayFixed_H
#define GEOMETRY_CALOGEOMETRY_EZArrayFixed_H 1

#include "Geometry/CaloGeometry/interface/EZMgr.h"

template < class T >
class EZArrayFixed
{
   public:

      typedef          EZMgr< T >               MgrType ;
      typedef typename MgrType::iterator        iterator ;
      typedef typename MgrType::const_iterator  const_iterator ;
      typedef typename MgrType::reference       reference ;
      typedef typename MgrType::const_reference const_reference ;
      typedef typename MgrType::size_type       size_type ;
      typedef typename MgrType::value_type      value_type ;

      EZArrayFixed< T >( const MgrType* mgr       , 
		    size_type      size      ,
		    const T&       t    = T()  ) :
	 m_begin ( 0==size ? (iterator)0 : mgr->assign( size, t ) ) ,
	 m_end   ( 0==size ? (iterator)0 : m_begin + size         ) ,
	 m_mgr   ( mgr )                                    { assert( !reallyEmpty() ) ; }

      EZArrayFixed< T >( const MgrType* mgr   , 
		    const_iterator start ,
		    const_iterator finis       ) :
	 m_begin ( 0==finis-start ? (iterator)0 : mgr->assign( finis - start ) ) ,
	 m_end   ( 0==finis-start ? (iterator)0 : m_begin    + finis - start   ) ,
	 m_mgr   ( mgr )
      {
	 assert( ( finis - start ) > 0 ) ;
	 iterator i ( begin() ) ;
	 for( const_iterator ic ( start ) ; ic != finis ; ++ic )
	 {
	    (*i) = (*ic) ;
	 }
      }

      virtual ~EZArrayFixed< T >() { m_mgr->release( m_begin ) ; }

      virtual const_iterator begin() const { return m_begin ; }
      virtual const_iterator end()   const { return m_end ; }

      virtual reference operator[]( const unsigned int i ) 
      {
	 iterator tmp ( m_begin + i ) ;
	 return *tmp ; 
      }

      virtual const_reference operator[]( const unsigned int i ) const 
      {
	 return (reference)(*this)[i] ;
      }

      bool reallyEmpty() const { return ( (iterator)0 == m_begin ) ; }
      
      virtual bool empty() const { return reallyEmpty() ; }

      virtual size_type size() const { return ( m_end - m_begin ) ; }

      virtual size_type capacity() const { return size() ; }

   protected:

      iterator startPtr() { return m_begin ; }

      void resizeSafe( size_type size ) { assert( reallyEmpty() ) ; m_mgr->assign( size ) ; }

      void assignSafe( size_type size, 
		       const T&  t = T() ) const
      { 
	 assert( reallyEmpty() ) ;
	 m_begin = m_mgr->assign( size, t ) ;
	 m_end   = m_begin + size ;
      }

      EZArrayFixed< T >( const MgrType* mgr      , 
		    size_type      size = 0 ,
		    const T&       t    = T() ,
		    int            dummy = 0    ) :         // this line to get unique signature
	 m_begin ( 0==size ? (iterator)0 : mgr->assign( size, t ) ) ,
	 m_end   ( 0==size ? (iterator)0 : m_begin + size         ) ,
	 m_mgr   ( mgr )   {}

   private:

      EZArrayFixed() ; //stop
      //EZArrayFixed( const EZArrayFixed& ) ; //stop
      //EZArrayFixed& operator=( const EZArrayFixed& ) ; //stop

      mutable iterator m_begin   ;
      mutable iterator m_end     ;
      const MgrType*   m_mgr   ;
};

#endif
