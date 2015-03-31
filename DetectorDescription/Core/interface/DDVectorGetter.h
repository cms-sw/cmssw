#ifndef CORE_DDVECTOR_GETTER_H
# define CORE_DDVECTOR_GETTER_H

# include "DetectorDescription/Core/interface/DDVector.h"
# include <string>
# include <vector>

namespace DDVectorGetter 
{
  bool		      	check( const std::string & );
  std::vector<double>   get( const std::string & );
  void                  beginWith( const std::string &, std::vector<std::string> & );

  inline bool
  check( const std::string & str )
  {
    DDVector::iterator<DDVector> vit;
    DDVector::iterator<DDVector> ved( DDVector::end());

    for(; vit != ved; ++vit )
    {
      if( vit->isDefined().second )
      {
	DDName vname( vit->name());
	if( vname.name() == str )
	  return true;
      }
    }

    return false;
  }

  inline std::vector<double>
  get( const std::string & str )
  {
    DDVector::iterator<DDVector> vit;
    DDVector::iterator<DDVector> ved( DDVector::end());
    if( vit == ved )
      throw cms::Exception( "DDException" ) << "DDVectorGetter: vectors are empty, cannot get array " << str;

    for (; vit != ved; ++vit )
    {
      if( vit->isDefined().second )
      {
	DDName vname( vit->name());
	if( vname.name() == str )
	{
	  const std::vector<double> & fvec = vit->values();
	  return fvec;
	}
      }
    }
  
    throw cms::Exception( "DDException" ) << "DDVectorGetter: cannot get array " << str;
  }

  inline void
  beginWith( const std::string & str, std::vector<std::string>& vstring )
  {
    DDVector::iterator<DDVector> vit;
    DDVector::iterator<DDVector> ved( DDVector::end());

    for(; vit != ved; ++vit )
    {
      if( vit->isDefined().second )
      {
	DDName vname( vit->name());
	if( vname.name().compare( 0, str.size(), str ) == 0 )
	  vstring.push_back( vname.name());
      }
    }
  }
};

#endif // CORE_DDVECTOR_GETTER_H
