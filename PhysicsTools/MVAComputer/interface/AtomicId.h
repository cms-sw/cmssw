#ifndef PhysicsTools_MVAComputer_AtomicId_h
#define PhysicsTools_MVAComputer_AtomicId_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     AtomicID
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: AtomicId.h,v 1.4 2007/12/08 16:11:10 saout Exp $
//

#include <ostream>
#include <string>

namespace PhysicsTools {

/** \class AtomicId
 *
 * \short Cheap generic unique keyword identifier class.
 *
 * AtomicId is a lightweight class intended to be used for key values
 * e.g. in STL maps. An atomic identifier can be transparently constructed
 * from and converted back to a string, but an instance of AtomicId
 * does not occupy and additional memory and the comparator operators are
 * very cheap. An AtomicId instance requires the size of a pointer and is
 * therefore suited for direct inlining.
 *
 ************************************************************/
class AtomicId {
    public:
	inline AtomicId() throw() : string(0) {}
	inline AtomicId(const AtomicId &orig) throw() : string(orig.string) {}
	/// constructs an AtomicId from a C string
	inline AtomicId(const char *arg) throw() : string(lookup(arg)) {}
	/// constructs an AtomicId from a STL string
	inline AtomicId(const std::string &arg) throw() :
		string(lookup(arg.c_str())) {}
	inline ~AtomicId() throw() {}

	inline AtomicId &operator = (const AtomicId &orig) throw()
	{ string = orig.string; return *this; }

	/// implicit cast to a C string
	inline operator const char *() const throw()
	{ return string; }

	/// null value check operator
	inline operator bool() const throw()
	{ return string != 0; }

	/// implicit cast to a STL string
	inline operator std::string() const throw()
	{ return std::string(string); }

	inline bool operator == (const AtomicId &second) const throw() { return string == second.string; }
	inline bool operator != (const AtomicId &second) const throw() { return string != second.string; }
	inline bool operator <  (const AtomicId &second) const throw() { return string <  second.string; }
	inline bool operator <= (const AtomicId &second) const throw() { return string <= second.string; }
	inline bool operator >  (const AtomicId &second) const throw() { return string >  second.string; }
	inline bool operator >= (const AtomicId &second) const throw() { return string >= second.string; }

    private:
	static AtomicId build(const char *arg) throw() { AtomicId q; q.string = arg; return q; }
	static const char *lookup(const char *arg) throw();

	const char	*string;
};

/// STL streaming operator
inline std::ostream &operator << (std::ostream &os, const PhysicsTools::AtomicId &id)
{ return os << (const char*)id; }

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_AtomicId_h
