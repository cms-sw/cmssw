#ifndef PhysicsTools_Discriminator_VarProcessor_h
#define PhysicsTools_Discriminator_VarProcessor_h
// -*- C++ -*-
//
// Package:     Discriminator
// Class  :     VarProcessor
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id$
//

#include <vector>

#include "PhysicsTools/MVAComputer/interface/ProcessRegistry.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/BitSet.h"

namespace PhysicsTools {

// forward declaration
class MVAComputer;

/** \class VarProcessor
 *
 * \short Common base class for variable processors.
 *
 * This base class for variable processors manages the common
 * interface to the global discriminator class and how the individual
 * variable processors are interconnected.
 *
 ************************************************************/
class VarProcessor :
	public ProcessRegistry<VarProcessor,
	                       Calibration::VarProcessor,
	                       const MVAComputer>::Factory {
    public:
	/** \class Config
	 *
	 * \short Helper class for discriminator computer set-up procedure
	 *
	 * This type is used to configure the variable processor and
	 * inter-processor passing of variables. The required variable
	 * flags are computed an the origin of multi-value variables tracked.
	 *
	 ************************************************************/
	struct Config {
		inline Config() : mask(Variable::FLAG_NONE), origin(0) {}
		inline Config(Variable::Flags mask, unsigned int origin) :
			mask(mask), origin(origin) {}

		Variable::Flags	mask;
		unsigned int	origin;
	};

	typedef std::vector<Config> Config_t;

	virtual ~VarProcessor();

	/// called from the discriminator computer to configure processor
	void configure(Config_t &config);

	/// run the processor evaluation pass on this processor
	void eval(double *input, int *conf,
	          double *output, int *outConf) const;

    protected:
	/** \class ConfIterator
	 *
	 * \short Iterator to loop over the input/output variable configuration
	 *
	 ************************************************************/
	struct ConfIterator {
	    public:
		/// apply property flags mask to variable at current position
		ConfIterator &operator () (Variable::Flags mask)
		{
			config[cur()].mask =
				(Variable::Flags)(config[cur()].mask & mask);
			return *this;
		}

		/// add a new output variable configuration \a config_
		ConfIterator &operator << (Config config_)
		{ config.push_back(config_); return *this; }

		/// add a new output variable configuration with mask \a mask
		ConfIterator &operator << (Variable::Flags mask)
		{ return *this << Config(mask, 0); }

		/// add a new output variable that inherits values from \a origin
		ConfIterator &operator << (const ConfIterator &origin)
		{ return *this << Config(config[origin.cur()].mask, origin.cur()); }

		/// test for end of iterator
		inline operator bool() const { return cur; }

		/// move to next input variable
		ConfIterator &operator ++ () { ++cur; return *this; }

		/// move to next input variable
		inline ConfIterator operator ++ (int dummy)
		{ ConfIterator orig = *this; operator ++ (); return orig; }

	    protected:
		friend class VarProcessor;

		ConfIterator(BitSet::Iterator cur, Config_t &config) :
			cur(cur), config(config) {}

	    private:
		BitSet::Iterator	cur;
		Config_t		&config;
	};

	/** \class ConfIterator
	 *
	 * \short Iterator to loop over the input/output variable values
	 *
	 ************************************************************/
	struct ValueIterator {
	    public:
		/// number of values for current input variable
		inline unsigned int size() const
		{ return conf[1] - conf[0]; }

		/// begin of value array for current input variable
		inline double *begin() const { return values; }

		/// end of value array for current input variable
		inline double *end() const { return values + size(); }

		/// the (first or only) value for the current input variable
		inline double operator * ()
		{ return *values; }

		/// value \a idx of current input variable
		inline double operator [] (unsigned int idx)
		{ return values[idx]; }

		/// add computed value to current output variable
		inline ValueIterator &operator << (double value)
		{ *output++ = value; outConf[1]++; return *this; }

		/// finish current output variable, move to next slot
		inline void operator () ()
		{ outConf++; outConf[1] = outConf[0]; }

		/// add \a value as output variable and move to next slot
		inline void operator () (double value)
		{ *this << value; (*this)(); }

		/// test for end of input variable iterator
		inline operator bool() const { return cur; }

		/// move to next input variable
		ValueIterator &operator ++ ()
		{
			BitSet::size_t orig = cur();
			unsigned int prev = *conf;
			conf += (++cur)() - orig; 
			values += *conf - prev;
			return *this;
		}

		/// move to next input variable
		inline ValueIterator operator ++ (int dummy)
		{ ValueIterator orig = *this; operator ++ (); return orig; }

	    protected:
		friend class VarProcessor;

		ValueIterator(BitSet::Iterator cur, double *values, int *conf,
		              double *output, int *outConf) :
			cur(cur), values(values), conf(conf),
			output(output), outConf(outConf)
		{
			outConf[1] = outConf[0];
			this->conf += cur();
			this->values += *this->conf;
		}

	    private:
		BitSet::Iterator	cur;
		double			*values;
		int			*conf;
		double			*output;
		int			*outConf;
	};

	typedef ProcessRegistry<VarProcessor,
		                Calibration::VarProcessor,
		                const MVAComputer> Registry;

	VarProcessor(const char *name, const Calibration::VarProcessor *calib,
	             const MVAComputer *computer);

	/// virtual configure method, implemented in actual processor
	virtual void configure(ConfIterator iter, unsigned int n) = 0;

	/// virtual evaluation method, implemented in actual processor
	virtual void eval(ValueIterator iter, unsigned int n) const = 0;

    protected:
	const MVAComputer	*computer;

    private:
	/// bit set to select the input variables to be passed to this processor
	BitSet			inputVars;
	unsigned int		nInputVars;
};

} // namespace PhysicsTools

#endif // PhysicsTools_Discriminator_VarProcessor_h
