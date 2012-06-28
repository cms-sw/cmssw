#ifndef Alignment_CommonAlignment_AlignableObjectId_h
#define Alignment_CommonAlignment_AlignableObjectId_h

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include <string>

/// Allows conversion between type and name, and vice-versa
// Obsolete. Use the two free functions below.
class AlignableObjectId 
{

public:
  AlignableObjectId(){};
  /// Convert name to type
  align::StructureType nameToType( const std::string &name ) const;

  /// Convert type to name
  std::string typeToName( align::StructureType type ) const;
};

const char *alignableObjecIdToString(align::StructureType type);
align::StructureType alignableObjectStringToId(const char *);

#endif
