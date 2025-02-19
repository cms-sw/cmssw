#ifndef Alignment_CommonAlignment_AlignableObjectId_h
#define Alignment_CommonAlignment_AlignableObjectId_h

#include <map>
#include <string>

#include "Alignment/CommonAlignment/interface/StructureType.h"

/// Allows conversion between type and name, and vice-versa
class AlignableObjectId 
{

public:
  
  /// Constructor (create maps)
  AlignableObjectId();

  typedef std::map<align::StructureType, std::string> MapEnumType;
  typedef std::map<std::string, align::StructureType> ReverseMapEnumType;
  typedef std::pair<align::StructureType, std::string> PairEnumType;
  typedef std::pair<std::string, align::StructureType> PairEnumReverseType;

  /// Convert name to type
  align::StructureType nameToType( const std::string& name ) const;

  /// Convert type to name
  const std::string& typeToName( align::StructureType type ) const;

private:
  MapEnumType theMap;
  ReverseMapEnumType theReverseMap;

  // Reverse functor
  struct reverse_pair {
	PairEnumReverseType operator()( const PairEnumType& pair ) const 
	{ 
	  return PairEnumReverseType( pair.second, pair.first ); 
	}
  };


};

#endif
