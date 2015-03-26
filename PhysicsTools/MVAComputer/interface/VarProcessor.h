#ifndef PhysicsTools_MVAComputer_VarProcessor_h
#define PhysicsTools_MVAComputer_VarProcessor_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     VarProcessor
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
//

#include <algorithm>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/ProcessRegistry.h"
#include "PhysicsTools/MVAComputer/interface/CalibrationFwd.h"
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

	class ConfigCtx {
	    public:
		typedef std::vector<Config>		Config_t;

		typedef Config_t::value_type		value_type;
		typedef Config_t::size_type		size_type;
		typedef Config_t::iterator		iterator;
		typedef Config_t::const_iterator	const_iterator;

		struct Context { virtual ~Context() {} };

		ConfigCtx(const std::vector<Variable::Flags>& flags);
		~ConfigCtx() { delete ctx; }

		inline size_type size() const { return configs.size(); }
		inline const_iterator begin() const { return configs.begin(); }
		inline iterator begin() { return configs.begin(); }
		inline const_iterator end() const { return configs.end(); }
		inline iterator end() { return configs.end(); }
		inline void push_back(const Config &config) { configs.push_back(config); }
		inline Config &operator [] (size_type i) { return configs[i]; }

	    private:		
		friend class VarProcessor;

		Config_t	configs;
		VarProcessor	*loop;
		Context		*ctx;
	};

        
	/** \class LoopCtx
	 *
	 * \short Hold context information for looping processors
	 *
	 * VarProcessor instances which allow looping need to keep
	 * track of the state of the loop between calls.
	 *
	 ************************************************************/
        class LoopCtx {
           public:
               LoopCtx(): index_(0), offset_(0), size_(0) {}
               inline unsigned int& index() { return index_;}
               inline unsigned int& offset() { return offset_;}
               inline unsigned int& size() { return size_;}
           private:
               unsigned int index_;
               unsigned int offset_;
               unsigned int size_ ;
        };

	virtual ~VarProcessor();

	/// called from the discriminator computer to configure processor
	void configure(ConfigCtx &config);

	/// run the processor evaluation pass on this processor
	inline void
	eval(double *input, int *conf, double *output, int *outConf,
	     int *loop, LoopCtx& loopCtx, unsigned int offset) const
	{
		ValueIterator iter(inputVars.iter(), input, conf,
		                   output, outConf, loop, loopCtx, offset);
		eval(iter, nInputVars);
	}

	/// run the processor evaluation pass on this processor and compute derivatives
	void deriv(double *input, int *conf, double *output, int *outConf,
	           int *loop, LoopCtx& ctx, unsigned int offset, unsigned int in,
	           unsigned int out, std::vector<double> &deriv) const;

	enum LoopStatus { kStop, kNext, kReset, kSkip };

	virtual LoopStatus loop(double *output, int *outConf,
	                        unsigned int nOutput,
                                LoopCtx& ctx,
	                        unsigned int &nOffset) const
	{ return kStop; }

   //used to create a PluginFactory
	struct Dummy {};
   typedef Dummy* PluginFunctionPrototype();

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

		/// return the current input variable flags
		Variable::Flags operator * () const
		{ return config[cur()].mask; }

		/// test for end of iterator
		inline operator bool() const { return cur; }

		/// move to next input variable
		ConfIterator &operator ++ () { ++cur; return *this; }

		/// move to next input variable
		inline ConfIterator operator ++ (int dummy)
		{ ConfIterator orig = *this; operator ++ (); return orig; }

	    protected:
		friend class VarProcessor;

		ConfIterator(BitSet::Iterator cur, ConfigCtx &config) :
			cur(cur), config(config) {}

	    private:
		BitSet::Iterator	cur;
		ConfigCtx		&config;
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
		{ return loop ? (conf[1] - conf[0]) : 1; }

		/// begin of value array for current input variable
		inline double *begin() const { return values; }

		/// end of value array for current input variable
		inline double *end() const { return values + size(); }

		/// checks for existence of values for current input variable
		inline bool empty() const { return begin() == end(); }

		/// the (first or only) value for the current input variable
		inline double operator * ()
		{ return *values; }

		/// value \a idx of current input variable
		inline double operator [] (unsigned int idx)
		{ return values[idx]; }

		/// add computed value to current output variable
		inline ValueIterator &operator << (double value)
		{ *output++ = value; return *this; }

		/// finish current output variable, move to next slot
		inline void operator () ()
		{
			int pos = output - start;
			if (*++outConf > pos)
				output = start + *outConf;
			else
				*outConf = pos;
		}

		/// add \a value as output variable and move to next slot
		inline void operator () (double value)
		{ *this << value; (*this)(); }

		/// test for end of input variable iterator
		inline operator bool() const { return cur; }

                inline LoopCtx& loopCtx() { return ctx;}

		/// move to next input variable
		ValueIterator &operator ++ ()
		{
			BitSet::size_t orig = cur();
			if (++cur) {
				unsigned int prev = *conf;
				conf += cur() - orig; 
				values += *conf - prev;
				if (loop && conf >= loop) {
					values += offset;
					loop = 0;
				}
			}
			return *this;
		}

		/// move to next input variable
		inline ValueIterator operator ++ (int dummy)
		{ ValueIterator orig = *this; operator ++ (); return orig; }

	    protected:
		friend class VarProcessor;

		ValueIterator(BitSet::Iterator cur, double *values,
		              int *conf, double *output, int *outConf,
		              int *loop, LoopCtx& ctx, unsigned int offset) :
                        cur(cur), ctx(ctx), offset(offset), start(values + offset),
			values(values), conf(conf), loop(loop),
			output(output + offset), outConf(outConf)
		{
			this->conf += cur();
			this->values += *this->conf;
			if (loop && this->conf >= loop) {
				this->values += offset;
				this->loop = 0;
			}
		}

	    private:
		BitSet::Iterator	cur;
                LoopCtx&                ctx;
		const unsigned int	offset;
		double			*const start;
		double			*values;
		const int		*conf;
		const int		*loop;
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

	/// virtual loop configure method
	virtual ConfigCtx::Context *
	configureLoop(ConfigCtx::Context *ctx, ConfigCtx::iterator begin,
	              ConfigCtx::iterator cur, ConfigCtx::iterator end);

	/// virtual evaluation method, implemented in actual processor
	virtual void eval(ValueIterator iter, unsigned int n) const = 0;

	/// virtual derivative evaluation method, implemented in actual processor
	virtual std::vector<double> deriv(ValueIterator iter,
	                                  unsigned int n) const
	{ return std::vector<double>(); }

    protected:
	const MVAComputer	*computer;

    private:
	/// bit set to select the input variables to be passed to this processor
	BitSet			inputVars;
	unsigned int		nInputVars;
};

template<>
VarProcessor *ProcessRegistry<VarProcessor, Calibration::VarProcessor,
                              const MVAComputer>::Factory::create(
	const char*, const Calibration::VarProcessor*, const MVAComputer*);

} // namespace PhysicsTools


#endif // PhysicsTools_MVAComputer_VarProcessor_h
