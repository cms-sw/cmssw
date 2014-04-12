// ----------------------------------------------------------------------
//
// ELstring.cc  Provides a string class with the semantics of std::string.
//              Customizers may substitute for this class to provide either
//              a string with a different allocator, or whatever else.
//
// The elements of string semantics which are relied upon are listed
// in doc/ELstring.semantics
//
// History:
//   15-Nov-2001  WEB  Inserted missing #include <cctype>
//
// ----------------------------------------------------------------------


#include "FWCore/MessageLogger/interface/ELstring.h"
#include <cctype>
#include <cstring>

namespace edm
{


bool  eq_nocase( const ELstring & s1, const char s2[] )  {
  using std::toupper;

  if (s1.length() != strlen(s2) ) return false;

  ELstring::const_iterator  p1;
  const char             *  p2;

  for ( p1 = s1.begin(), p2 = s2;  *p2 != '\0';  ++p1, ++p2 )  {
    if ( toupper(*p1) != toupper(*p2) )  {
      return false;
    }
  }
  return true;

}  // eq_nocase()


bool  eq( const ELstring & s1, const ELstring s2 )  {

  return  s1 == s2;

}  // eq()


} // end of namespace edm  */
