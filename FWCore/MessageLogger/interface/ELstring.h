#ifndef MessageLogger_ELstring_h
#define MessageLogger_ELstring_h


// ----------------------------------------------------------------------
//
// ELstring.h	Provides a string class with the semantics of std::string.
// 		Customizers may substitute for this class to provide either
//		a string with a different allocator, or whatever else.
//
// The elements of string semantics which are relied upon are listed
// in ELstring.semantics
//
// ----------------------------------------------------------------------


#include <string>


namespace edm {       


// ----------------------------------------------------------------------


typedef std::string ELstring;

bool eq_nocase( const ELstring & s1, const char s2[] );

bool eq( const ELstring & s1, const ELstring s2 );


// ----------------------------------------------------------------------


}        // end of namespace edm


#endif  // MessageLogger_ELstring_h
