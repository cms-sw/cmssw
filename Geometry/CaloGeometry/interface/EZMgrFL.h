#ifndef GEOMETRY_CALOGEOMETRY_EZMGRFL_H
#define GEOMETRY_CALOGEOMETRY_EZMGRFL_H 1

#include <vector>
#include <assert.h>

template < class T >
class EZMgrFL
{
   public:

      typedef std::vector<T>                    VecType ;
      typedef typename VecType::iterator        iterator ;
      typedef typename VecType::const_iterator  const_iterator ;
      typedef typename VecType::reference       reference ;
      typedef typename VecType::const_reference const_reference ;
      typedef typename VecType::value_type      value_type ;
      typedef typename VecType::size_type       size_type ;

      EZMgrFL< T >( size_type vecSize ,
		    size_type subSize   ) : m_vecSize ( vecSize ) ,
					    m_subSize ( subSize ) ,
					    m_counter ( 0 )
      {
	 m_vec.resize(0); 
	 assert( subSize > 0 ) ;
	 assert( vecSize > 0 ) ;
	 assert( 0 == m_vec.capacity() ) ;
      }

      virtual ~EZMgrFL< T >() { assert( 0 == m_counter ) ; } 

      iterator reserve( size_type size = m_subSize ) const { return assign() ; }
      iterator resize(  size_type size = m_subSize ) const { return assign() ; }

      iterator assign( const T& t = T() ) const
      {
	 assert( ( m_vec.size() + size ) <= m_vecSize ) ;
	 if( 0 == m_vec.capacity() )
	 {
	    m_vec.reserve( m_vecSize ) ;
	    assert( m_vecSize == m_vec.capacity() ) ;
	 }
	 iterator start ( m_vec.end() ) ;
	 for( size_type  i ( 0 ) ; i != m_subSize ; ++i )
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

      size_type subSize() const { return m_subSize ; }

   private:

      EZMgrFL() ; //stop
      EZMgrFL( const EZMgrFL& ) ; //stop
      EZMgrFL& operator=( const EZMgrFL& ) ; //stop

      const size_type m_vecSize ;
      const size_type m_subSize ;
      mutable VecType m_vec     ;
      mutable unsigned int m_counter ;
};

#endif
