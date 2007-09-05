#ifndef GEOMETRY_CALOGEOMETRY_EZArray_H
#define GEOMETRY_CALOGEOMETRY_EZArray_H 1

#include "Geometry/CaloGeometry/interface/EZArrayFixed.h"

template < class T >
class EZArray : public EZArrayFixed< T >
{
   public:

      typedef          EZArrayFixed< T >        ParentType ;
      typedef typename ParentType::MgrType      MgrType ;
      typedef typename MgrType::iterator        iterator ;
      typedef typename MgrType::const_iterator  const_iterator ;
      typedef typename MgrType::reference       reference ;
      typedef typename MgrType::const_reference const_reference ;
      typedef typename MgrType::size_type       size_type ;
      typedef typename MgrType::value_type      value_type ;

      EZArray< T >( const MgrType* mgr       , 
		    size_type      size = 0  ,
		    const T&       t    = T()  ) : ParentType( mgr, size, t, (int)0 ) {}

      EZArray< T >( const MgrType* mgr   , 
			const_iterator start ,
			const_iterator finis       ) : ParentType( mgr, start, finis )  {}

      virtual ~EZArray< T >() {}

      virtual void resize( size_type size ) { resizeSafe( size ) ; }

      virtual void assign( size_type size,
			   const T&  t = T() ) const { assignSafe( size, t ) ; }

      virtual const_iterator begin() const 
      {
	 return ParentType::begin() ; 
      }

      virtual reference operator[]( const unsigned int i ) 
      {
	 return *( ParentType::startPtr() + i ) ; 
      }

      virtual const_reference operator[]( const unsigned int i ) const 
      {
	 return (reference)(*this)[i] ;
      }

   protected:

   private:

      EZArray< T >() ; //stop
      //EZArray( const EZArray& ) ; //stop
      //EZArray& operator=( const EZArray& ) ; //stop
};

#endif
