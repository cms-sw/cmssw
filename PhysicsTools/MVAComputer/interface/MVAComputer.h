#ifndef PhysicsTools_MVAComputer_MVAComputer_h
#define PhysicsTools_MVAComputer_MVAComputer_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     MVAComputer
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: MVAComputer.h,v 1.7 2009/06/03 09:50:14 saout Exp $
//

#include <iostream>
#include <vector>
#include <memory>

#include "PhysicsTools/MVAComputer/interface/CalibrationFwd.h"
#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

namespace PhysicsTools {

/** \class MVAComputer
 *
 * \short Main interface class to the generic discriminator computer framework.
 *
 * The MVAComputer class represents an instance of the modular
 * discriminator computer. It is constructed from a "calibration" object
 * which contains all the required histograms, matrices and other trainina
 * data required for computing the discriminator. The calibration data also
 * defines the names and properties of variables that can passed to that
 * instance of the discriminator computer. The evaluation methods then
 * calculates the discriminator from a applicable set of input variables,
 * i.e. vector of key-value pairs.
 *
 ************************************************************/
class MVAComputer {
    public:
	/// construct a discriminator computer from a const calibation object
	MVAComputer(const Calibration::MVAComputer *calib);

	/// construct a discriminator computer from a calibation object
	MVAComputer(Calibration::MVAComputer *calib, bool owned = false);

	~MVAComputer();

	/// evaluate variables given by a range of iterators given by \a first and \a last
	template<typename Iterator_t>
	double eval(Iterator_t first, Iterator_t last) const;

	template<typename Iterator_t>
	double deriv(Iterator_t first, Iterator_t last) const;

	/// evaluate variables in iterable container \a values
	template<typename Container_t>
	inline double eval(const Container_t &values) const
	{
		typedef typename Container_t::const_iterator Iterator_t;
		return this->template eval<Iterator_t>(
						values.begin(), values.end());
	}

	template<typename Container_t>
	inline double deriv(Container_t &values) const
	{
		typedef typename Container_t::iterator Iterator_t;
		return this->template deriv<Iterator_t>(
						values.begin(), values.end());
	}

	/* various methods for standalone use of calibration files */

	/// read calibration object from plain file
	static Calibration::MVAComputer *readCalibration(const char *filename);

	/// read calibration object from plain C++ input stream
	static Calibration::MVAComputer *readCalibration(std::istream &is);

	/// write calibration object to file
	static void writeCalibration(const char *filename,
	                             const Calibration::MVAComputer *calib);

	/// write calibration object to pain C++ output stream
	static void writeCalibration(std::ostream &os,
	                             const Calibration::MVAComputer *calib);

	/// construct a discriminator computer from a calibration file
	MVAComputer(const char *filename);

	/// construct a discriminator computer from C++ input stream
	MVAComputer(std::istream &is);

    private:
	/** \class InputVar
	 * \short input variable configuration object
	 */
	struct InputVar {
		/// generic variable information (name, ...)
		Variable	var; 

		/// variable index in fixed-position evaluation array
		unsigned int	index;

		/// number of times each appearance of that variable can appear while computing the discriminator
		unsigned int	multiplicity;

		bool operator < (AtomicId id) const
		{ return var.getName() < id; }

		bool operator < (const InputVar &other) const
		{ return var.getName() < other.var.getName(); }
	};

	/** \class Processor
	 * \short variable processor container
	 */
	struct Processor {
		inline Processor(VarProcessor *processor,
		                 unsigned int nOutput) :
			processor(processor), nOutput(nOutput) {}

		inline Processor(const Processor &orig)
		{ processor = orig.processor; nOutput = orig.nOutput; }

		inline Processor &operator = (const Processor &orig)
		{ processor = orig.processor; nOutput = orig.nOutput; return *this; }

		/// owned variable processor instance
		mutable std::auto_ptr<VarProcessor>	processor;

		/// number of output variables
		unsigned int				nOutput;
	};

	struct EvalContext {
		EvalContext(double *values, int *conf, unsigned int n) :
			values_(values), conf_(conf), n_(n) {}

		inline void eval(const VarProcessor *proc, int *outConf,
		                 double *output, int *loop,
		                 unsigned int offset, unsigned int out) const
		{ proc->eval(values_, conf_, output, outConf, loop, offset); }

		inline double output(unsigned int output) const
		{ return values_[conf_[output]]; }

		inline double *values() const { return values_; }
		inline int *conf() const { return conf_; }
		inline unsigned int n() const { return n_; }

		double		*values_;
		int		*conf_;
		unsigned int	n_;
	};

	struct DerivContext {
		DerivContext() : n_(0) {}

		void eval(const VarProcessor *proc, int *outConf,
		          double *output, int *loop,
		          unsigned int offset, unsigned int out) const;

		double output(unsigned int output,
		              std::vector<double> &derivs) const;

		inline double *values() const { return &values_.front(); }
		inline int *conf() const { return &conf_.front(); }
		inline unsigned int n() const { return n_; }

		mutable std::vector<double>	values_;
		mutable std::vector<double>	deriv_;
		mutable std::vector<int>	conf_;
		unsigned int			n_;
	};
	
	/// construct processors from calibration and setup variables
	void setup(const Calibration::MVAComputer *calib);

	/// map variable identifier \a name to the numerical position in the array
	unsigned int getVariableId(AtomicId name) const;

	/// evaluate discriminator from flattened variable array
	template<class T> void evalInternal(T &ctx) const;

	/// vector of input variables
	std::vector<InputVar>	inputVariables;

	/// vector of variable processors
	std::vector<Processor>	varProcessors;

	/// total number of variables to expect while computing discriminator
	unsigned int		nVars;

	/// index of the variable in the "conf" array to return as result
	unsigned int		output;

	/// in case calibration object is owned by the MVAComputer
	std::auto_ptr<Calibration::MVAComputer> owned;
};

} // namespace PhysicsTools

#include "PhysicsTools/MVAComputer/interface/MVAComputer.icc"

#endif // PhysicsTools_MVAComputer_MVAComputer_h
