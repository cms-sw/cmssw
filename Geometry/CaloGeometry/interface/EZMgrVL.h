#ifndef GEOMETRY_CALOGEOMETRY_EZMGRVL_H
#define GEOMETRY_CALOGEOMETRY_EZMGRVL_H 1

#include <vector>
#include <assert.h>

template < class T >
class EZMgrVL
{
   public:

      typedef std::vector<T>                    VecType ;
      typedef typename VecType::iterator        iterator ;
      typedef typename VecType::const_iterator  const_iterator ;
      typedef typename VecType::reference       reference ;
      typedef typename VecType::const_reference const_reference ;
      typedef typename VecType::value_type      value_type ;
      typedef typename VecType::size_type       size_type ;

      EZMgrVL< T >( size_type vecSize ) : m_vecSize ( vecSize ) ,
					m_counter ( 0 )
      {
	 m_vec.resize(0); 
	 assert( m_vec.capacity() == 0 ) ;
      }

      virtual ~EZMgrVL< T >() { assert( 0 == m_counter ) ; } 

      iterator reserve( size_type size ) const { return assign( size ) ; }
      iterator resize(  size_type size ) const { return assign( size ) ; }

      iterator assign( size_type size, const T& t = T() ) const
      {
	 assert( ( m_vec.size() + size ) <= m_vecSize ) ;
	 if( 0 == m_vec.capacity() )
	 {
	    m_vec.reserve( m_vecSize ) ;
	    assert( m_vecSize == m_vec.capacity() ) ;
	 }
	 iterator start ( m_vec.end() ) ;
	 for( size_type  i ( 0 ) ; i != size ; ++i )
	 {
	    m_vec.push_back( t ) ;
	 }
	 ++m_counter ;
	 return start ;
      }

      void release( iterator it ) const 
      {
	 assert( 0 != m_counter ) ;
	 if( 0 == --m_counter ) 
	 {
	    m_vec.erase( m_vec.begin(), m_vec.end() ) ;
	 }
      }

   private:

      EZMgrVL() ; //stop
      EZMgrVL( const EZMgrVL& ) ; //stop
      EZMgrVL& operator=( const EZMgrVL& ) ; //stop

      const size_type m_vecSize ;
      mutable VecType m_vec     ;
      mutable unsigned int m_counter ;
};

#endif
