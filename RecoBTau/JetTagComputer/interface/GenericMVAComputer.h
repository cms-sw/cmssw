#ifndef RecoBTau_BTauComputer_GenericMVAComputer_h
#define RecoBTau_BTauComputer_GenericMVAComputer_h

#include <iterator>
#include <vector>

#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

// overload MVAComputer and replace eval() methods to work on TaggingVariable
class GenericMVAComputer : public PhysicsTools::MVAComputer {
    public:
	// forward declarations;
	template<typename Iter_t> class TaggingVariableIterator;
	class TaggingVariableMapping;

	GenericMVAComputer(const PhysicsTools::Calibration::MVAComputer *calib) :
		PhysicsTools::MVAComputer(calib) {}

	// create wrapping iterator
	template<typename Iter_t>
	inline TaggingVariableIterator<Iter_t> iterator(Iter_t iter) const
	{ return TaggingVariableIterator<Iter_t>(&mapping, iter); }

	// overload eval method to work on containers of TaggingVariable
	template<typename Iter_t>
	inline double eval(Iter_t first, Iter_t last) const
	{
		typedef TaggingVariableIterator<Iter_t> Wrapped_t;
		return PhysicsTools::MVAComputer::template eval<Wrapped_t>(
			iterator<Iter_t>(first), iterator<Iter_t>(last));
	}

	template<typename Container_t>
	inline double eval(const Container_t &values) const
	{
		typedef typename Container_t::const_iterator Iter_t;
		return this->template eval<Iter_t>(values.begin(), values.end());
	}

	// iterator wrapper with on-the-fly TaggingVariableName -> AtomicId mapping
	//
	// Notice that this tells the computer to completely inline it
	// this is reasonable since inlining it means that the optimizer
	// will not produce any additional code for the wrapper
	// (it is entirely optimized away), except for the actual mapping
	// which is no more than a simple array lookup.
	//
	// The result will be inlined into the eval() template of MVAComputer
	// which will be instantiated as one single tightly-integrated
	// function.
	template<typename Iter_t>
	class TaggingVariableIterator {
	    public:
		// class implementing MVAComputer::Variable::Value interface
		struct Value {
		    public:
			// the actual operator doing the mapping
			inline PhysicsTools::AtomicId getName() const
			{ return mapping->getAtomicId(iter->first); }

			inline double getValue() const
			{ return iter->second; }

			operator PhysicsTools::Variable::Value() const
			{
				return PhysicsTools::Variable::Value(
						getName(), getValue());
			}

		    protected:
			friend class TaggingVariableIterator;

			inline Value(TaggingVariableMapping *mapping,
			             const Iter_t &iter) :
				mapping(mapping), iter(iter) {}

		    private:
			// pointer to the current mapping
			TaggingVariableMapping	*mapping;
			// iterator to reco::TaggingVariable in orig. container
			Iter_t			iter;
		};

		typedef std::forward_iterator_tag				iterator_category;
		typedef Value							value_type;
		typedef typename std::iterator_traits<Iter_t>::difference_type	difference_type;
		typedef const Value						*pointer;
		typedef const Value						&reference;

		inline ~TaggingVariableIterator() {}

		// methods to make class a standard forward iterator

		inline const Value &operator * () const { return value; }
		inline Value &operator * () { return value; }

		inline const Value *operator -> () const { return &value; }
		inline Value *operator -> () { return &value; }

		inline bool operator == (const TaggingVariableIterator &other) const
		{ return value.iter == other.value.iter; }
		inline bool operator != (const TaggingVariableIterator &other) const
		{ return value.iter != other.value.iter; }
		inline bool operator < (const TaggingVariableIterator &other) const
		{ return value.iter < other.value.iter; }

		inline TaggingVariableIterator &operator ++ ()
		{ ++value.iter; return *this; }

		inline TaggingVariableIterator operator ++ (int dummy)
		{ TaggingVariableIterator orig = *this; ++value.iter; return orig; }

	    protected:
		friend class GenericMVAComputer;
		inline TaggingVariableIterator(TaggingVariableMapping *mapping,
		                               const Iter_t &iter) :
			value(mapping, iter) {}

	    private:
		// holds the current "value"
		// it's really only an iterator that points the original
		// current TaggingVariable plus required methods
		Value			value;
	};

	// TaggingVariableName -> PhysicsTools::AtomicId mapping
	class TaggingVariableMapping {
	    public:
		typedef PhysicsTools::AtomicId		AtomicId;
		typedef reco::TaggingVariableName	TaggingName;

		TaggingVariableMapping();
		~TaggingVariableMapping() {}

		inline AtomicId getAtomicId(TaggingName taggingName) const
		{ return taggingVarToAtomicId[taggingName]; }

	    private:
		std::vector<AtomicId>	taggingVarToAtomicId;
	};

    private:
	static TaggingVariableMapping	mapping;
};

#endif // RecoBTau_BTauComputer_GenericMVAComputer_h
