#ifndef PhysicsTools_MVAComputer_Variable_h
#define PhysicsTools_MVAComputer_Variable_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     Variable
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: Variable.h,v 1.3 2007/05/25 16:37:58 saout Exp $
//

#include <string>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

namespace PhysicsTools {

/** \class Variable
 *
 * \short Class describing an input variable.
 *
 * The class Variable describes an input variable by its name and properties.
 * The name is represented by an atomic identifier (an alphanumerical value).
 * The properties consists of flags. Currently the two flags describe
 * the allowed number of times a variable can appear in the input, whether
 * a variable is allowed to be omitted and whether a variable can have
 * multiple values.
 *
 ************************************************************/
class Variable {
    public:
	enum Flags {
		FLAG_NONE	= 0,
		FLAG_OPTIONAL	= 1 << 0,
		FLAG_MULTIPLE	= 1 << 1,
		FLAG_ALL	= (1 << 2) - 1
	};

	/** \class Value
	 *
	 * \short Helper class that can contain an identifier-value pair
	 *
	 * Variable::Value contains an instance of an input variable that
	 * is identified by the atomic identifer of the variable and carries
	 * an associated double value.
	 *
	 ************************************************************/
	struct Value {
		inline Value() {}
		inline Value(AtomicId name, double value) :
			name(name), value(value) {}

		inline Value &operator = (const Value &orig)
		{ name = orig.name; value = orig.value; return *this; }

		inline AtomicId getName() const { return name; }
		inline double getValue() const { return value; }

		AtomicId	name;
		double		value;
	};

	inline Variable() {}
	inline Variable(AtomicId name, Flags flags = FLAG_NONE) :
		name(name), flags(flags) {}

	const AtomicId getName() const
	{ return name; }

	bool isOptional() const
	{ return flags & FLAG_OPTIONAL; }

	bool isMultiple() const
	{ return flags & FLAG_MULTIPLE; }

    private:
	AtomicId	name;
	Flags		flags;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_Variable_h
