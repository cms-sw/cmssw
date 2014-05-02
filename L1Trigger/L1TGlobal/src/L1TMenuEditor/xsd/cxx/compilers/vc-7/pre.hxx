// file      : xsd/cxx/compilers/vc-7/pre.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file


#if (_MSC_VER < 1310)
#  error Microsoft Visual C++ 7.0 (.NET 2002) is not supported.
#endif


// These warnings had to be disabled "for good".
//
#pragma warning (disable:4250) // inherits via dominance
#pragma warning (disable:4505) // unreferenced local function has been removed
#pragma warning (disable:4661) // no definition for explicit instantiation


// Push warning state.
//
#pragma warning (push, 3)


// Disabled warnings.
//
#pragma warning (disable:4355) // passing 'this' to a member
#pragma warning (disable:4584) // is already a base-class
#pragma warning (disable:4800) // forcing value to bool
#pragma warning (disable:4275) // non dll-interface base
#pragma warning (disable:4251) // base needs to have dll-interface
#pragma warning (disable:4224) // nonstandard extension (/Za option)


// Elevated warnings.
//
#pragma warning (2:4239) // standard doesn't allow this conversion
