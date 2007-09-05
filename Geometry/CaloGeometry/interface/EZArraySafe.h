#ifndef GEOMETRY_CALOGEOMETRY_EZArraySafe_H
#define GEOMETRY_CALOGEOMETRY_EZArraySafe_H 1

#include "Geometry/CaloGeometry/interface/EZArray.h"

template < class T >
class EZArraySafe : public EZArray< T >
{
   public:

      typedef          EZArray< T >             ParentType ;
      typedef typename ParentType::MgrType      MgrType ;
      typedef typename MgrType::iterator        iterator ;
      typedef typename MgrType::const_iterator  const_iterator ;
      typedef typename MgrType::reference       reference ;
      typedef typename MgrType::const_reference const_reference ;
      typedef typename MgrType::size_type       size_type ;
      typedef typename MgrType::value_type      value_type ;

      EZArraySafe< T >( const MgrType* mgr       , 
			size_type      size = 0  ,
			const T&       t    = T()  ) : ParentType( mgr, size, t ) {}

      EZArraySafe< T >( const MgrType* mgr   , 
			const_iterator start ,
			const_iterator finis       ) : ParentType( mgr, start, finis )  {}

      virtual ~EZArraySafe< T >() {}

      virtual const_iterator begin() const 
      {
	 assert( !ParentType::reallyEmpty() ) ; // safe
	 return ParentType::begin() ; 
      }

      virtual reference operator[]( const unsigned int i ) 
      {
	 assert( !ParentType::reallyEmpty() ) ; //safe
	 assert( i < ParentType::size() ) ; //safe
	 return *( ParentType::startPtr() + i ) ; 
      }

      virtual const_reference operator[]( const unsigned int i ) const 
      {
	 assert( i < ParentType::size() ) ; // safe
	 return (reference)(*this)[i] ;
      }

   protected:

   private:

      EZArraySafe< T >() ; //stop
      //EZArraySafe( const EZArray& ) ; //stop
      //EZArraySafe& operator=( const EZArray& ) ; //stop
};

#endif
