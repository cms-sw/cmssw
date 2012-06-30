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
  static const char *idToString(align::StructureType type);
  static align::StructureType stringToId(const char *);
  static align::StructureType stringToId(const std::string &s) { return stringToId(s.c_str()); };
};


#endif
