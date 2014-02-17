#ifndef PhysicsTools_MVAComputer_BitSet_h
#define PhysicsTools_MVAComputer_BitSet_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     BitSet
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: BitSet.h,v 1.5 2009/05/09 12:23:46 saout Exp $
//

#include <string.h>
#include <cstddef>
#include <cstring>

namespace PhysicsTools {

/** \class BitSet
 *
 * \short A compact container for storing single bits.
 *
 * BitSet provides a container of boolean values, similar to
 * a std::vector<bool> which only consumes one actual bit for each value.
 * Also an iterator is provided that can iterate over all set bits.
 *
 ************************************************************/
class BitSet {
    public:
	typedef std::size_t size_t;

    protected:
	typedef unsigned int Word_t;

    public:
	/** \class Manipulator
	 *
	 * \short Opaque structure for transparent write access to individual bits.
	 *
	 * This structure is used transparently on the left side of an
	 * assigment operator when writing a boolean value to an
	 * individual bit, i.e. represents some sort of bit reference type.
	 *
	 ************************************************************/
	struct Manipulator {
	    public:
		inline Manipulator(const Manipulator &orig) :
			word(orig.word), mask(orig.mask) {}
		inline ~Manipulator() {}

		/// implicit cast to pointed-at boolean bit value
		inline operator bool() const { return *word & mask; }

		/// bit assignment operator
		inline bool operator = (bool bit)
		{ *word = (*word & ~mask) | (bit ? mask : 0); return bit; }

	    protected:
		friend class BitSet;

		inline Manipulator(Word_t *word, unsigned int bit) :
			word(word), mask((Word_t)1 << bit) {}

	    private:
		Word_t		*word;
		Word_t		mask;
	};

	/** \class Iterator
	 *
	 * \short Iterates over all set bits of a BitSet.
	 *
	 * This structure is used to iterate over all set bits in a BitSet.
	 *
	 ************************************************************/
	struct Iterator {
	    public:
		/// boolean test for the end of the BitSet
		inline operator bool() const { return store < end; }

		/// returns the index of the currently pointed-at bit
		inline size_t operator () () const
		{ return (store - begin) * wordSize + pos; }

		/// increment iterator to point at the next set bit
		Iterator &operator ++ ()
		{
			if (++pos < wordSize) {
				Word_t word = *store & -(1 << pos);
				if (word) {
					pos = ffs(word) - 1;
					return *this;
				}
			}

			pos = 0;
			for(;;) {
				if (++store >= end)
					break;
				else if (*store) {
					pos = ffs(*store) - 1;
					break;
				}
			}

			return *this;
		}

		/// increment iterator to point at the next set bit
		inline Iterator operator ++ (int dummy)
		{ Iterator orig = *this; ++*this; return orig; }

	    protected:
		friend class BitSet;

		Iterator(Word_t *begin, Word_t *end) :
			begin(begin), store(begin), end(end), pos(0)
		{ if (store < end && !(*store & 1)) ++*this; }

	    private:
		Word_t		*begin, *store, *end;
		unsigned int	pos;
	};

	BitSet() : store(0), bits_(0) {}

	BitSet(const BitSet &orig) : bits_(orig.bits_)
	{
		std::size_t words = bitsToWords(bits_);
		if (words) {
			store = new Word_t[words];
			std::memcpy(store, orig.store, words * sizeof(Word_t));
		} else
			store = 0;
	}

	/// construct BitSet with a fixed size of \a bits bits
	BitSet(size_t bits) : bits_(bits)
	{
		std::size_t words = bitsToWords(bits);
		if (words) {
			store = new Word_t[words];
			std::memset(store, 0, sizeof(Word_t) * words);
		} else
			store = 0;
	}

	inline ~BitSet() { delete[] store; }

	BitSet &operator = (const BitSet &orig)
	{
		delete[] store;
		bits_ = orig.bits_;
		std::size_t words = bitsToWords(bits_);
		if (words) {
			store = new Word_t[words];
			std::memcpy(store, orig.store, words * sizeof(Word_t));
		} else
			store = 0;
		return *this;
	}

	/// provide read/write access to bit with index \a bit via reference
	inline Manipulator operator [] (size_t bit)
	{ return Manipulator(&store[bit / wordSize], bit % wordSize); }

	/// provide read access to bit with index \a bit via reference
	inline const Manipulator operator [] (size_t bit) const
	{ return Manipulator(&store[bit / wordSize], bit % wordSize); }

	/// returns the number of all bits in the container
	inline size_t size() const { return bits_; }

	/// returns the number of set bits in the container
	size_t bits() const;

	/// create iterator over all set bits
	inline Iterator iter() const
	{ return Iterator(store, store + bitsToWords(bits_)); }

    private:
	static inline size_t bitsToWords(std::size_t bits)
	{ return (bits + wordSize - 1) / wordSize; }

	static const unsigned int wordSize = sizeof(Word_t) * 8;

	Word_t	*store;
	size_t	bits_;
};

namespace Calibration {
	class BitSet;

	/// constructs BitSet container from persistent representation
	PhysicsTools::BitSet convert(const BitSet &bitSet);
	/// convert BitSet container into persistent representation
	BitSet convert(const PhysicsTools::BitSet &bitSet);
}

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_BitSet_h
