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
// $Id: Variable.h,v 1.8 2009/03/27 14:33:38 saout Exp $
//

#include <vector>
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
	class Value {
	    public:
		inline Value() {}
		inline Value(const Value &orig) :
			name(orig.name), value(orig.value) {}
		inline Value(AtomicId name, double value) :
			name(name), value(value) {}

		inline Value &operator = (const Value &orig)
		{ name = orig.name; value = orig.value; return *this; }

		inline void setName(AtomicId name) { this->name = name; }
		inline void setValue(double value) { this->value = value; }

		inline AtomicId getName() const { return name; }
		inline double getValue() const { return value; }

	    private:
		AtomicId	name;
		double		value;
	};

	/** \class ValueList
	 *
	 * \short Helper class that can contain an list of identifier-value pairs
	 *
	 * Variable::ValueList contains a vector of Variable::Value with
	 * additional convenience methods.
	 *
	 ************************************************************/
	class ValueList {
	    public:
		typedef std::vector<Value>	_Data;
		typedef _Data::value_type	value_type;
		typedef _Data::pointer		pointer;
		typedef _Data::const_pointer	const_pointer;
		typedef _Data::reference	reference;
		typedef _Data::const_reference	const_reference;
		typedef _Data::iterator		iterator;
		typedef _Data::const_iterator	const_iterator;
		typedef _Data::size_type	size_type;
		typedef _Data::difference_type	difference_type;
		typedef _Data::allocator_type	allocator_type;

		inline ValueList() {}
		inline ValueList(const ValueList &orig) : data_(orig.data_) {}
		inline ValueList(const _Data &orig) : data_(orig) {}
		inline ~ValueList() {}

		inline ValueList &operator = (const ValueList &orig)
		{ data_ = orig.data_; return *this; }

		inline void clear()
		{ data_.clear(); }

		inline void add(AtomicId id, double value)
		{ data_.push_back(Value(id, value)); }

		inline void add(const Value &value)
		{ data_.push_back(value); }

		inline size_type size() const { return data_.size(); }
		bool empty() const { return data_.empty(); }

		inline const_iterator begin() const { return data_.begin(); }
		iterator begin() { return data_.begin(); }

		inline const_iterator end() const { return data_.end(); }
		iterator end() { return data_.end(); }

		inline const _Data &values() const { return data_; }
		_Data &values() { return data_; }

		inline const_reference front() const { return *data_.begin(); }
		reference front() { return *data_.begin(); }

		inline const_reference back() const { return *data_.rbegin(); }
		reference back() { return *data_.rbegin(); }

		inline const_pointer data() const { return &front(); }
		pointer data() { return &front(); }

	    private:
		std::vector<Value>	data_;
	};

	inline Variable() {}
	inline Variable(const Variable &orig) :
		name(orig.name), flags(orig.flags) {}
	inline Variable(AtomicId name, Flags flags = FLAG_NONE) :
		name(name), flags(flags) {}

	const AtomicId getName() const { return name; }
	Flags getFlags() const { return flags;}

	bool isOptional() const	{ return flags & FLAG_OPTIONAL; }
	bool isMultiple() const { return flags & FLAG_MULTIPLE; }

    private:
	AtomicId	name;
	Flags		flags;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_Variable_h
