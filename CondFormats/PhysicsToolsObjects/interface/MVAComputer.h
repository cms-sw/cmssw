#ifndef CondFormats_PhysicsToolsObjects_MVAComputer_h
#define CondFormats_PhysicsToolsObjects_MVAComputer_h
// -*- C++ -*-
//
// Package:     PhysicsToolsObjects
// Class  :     MVAComputer
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: MVAComputer.h,v 1.15 2010/01/26 19:40:03 saout Exp $
//

#include "CondFormats/Serialization/interface/Serializable.h"

#include <memory>
#include <string>
#include <vector>
#include <map>

#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"

namespace PhysicsTools {
namespace Calibration {

// helper classes

class BitSet {
    public:
	// help that poor ROOT to copy bitsets... (workaround)
	BitSet &operator = (const BitSet &other)
	{ store = other.store; bitsInLast = other.bitsInLast; return *this; }

	std::vector<unsigned char>	store;
	unsigned int			bitsInLast;

  COND_SERIALIZABLE;
};

class Matrix {
    public:
	std::vector<double>		elements;
	unsigned int			rows;
	unsigned int			columns;

  COND_SERIALIZABLE;
};

// configuration base classes

class VarProcessor {
    public:
	BitSet	inputVars;

	virtual ~VarProcessor() {}
	virtual std::string getInstanceName() const;
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif

  COND_SERIALIZABLE;
};

class Variable {
    public:
	inline Variable() {}
	inline Variable(const std::string &name) : name(name) {}
	inline ~Variable() {}

	std::string			name;

  COND_SERIALIZABLE;
};

// variable processors

class ProcOptional : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	std::vector<double>		neutralPos;

  COND_SERIALIZABLE;
};

class ProcCount : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
  COND_SERIALIZABLE;
};

class ProcClassed : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	unsigned int			nClasses;

  COND_SERIALIZABLE;
};

class ProcSplitter : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	unsigned int			nFirst;

  COND_SERIALIZABLE;
};

class ProcForeach : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	unsigned int			nProcs;

  COND_SERIALIZABLE;
};

class ProcSort : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	unsigned int			sortByIndex;
	bool				descending;

  COND_SERIALIZABLE;
};

class ProcCategory : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	typedef std::vector<double> BinLimits;

	std::vector<BinLimits>		variableBinLimits;
	std::vector<int>		categoryMapping;

  COND_SERIALIZABLE;
};

class ProcNormalize : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	std::vector<HistogramF>		distr;
	int				categoryIdx;

  COND_SERIALIZABLE;
};

class ProcLikelihood : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	class SigBkg {
	    public:
		HistogramF		background;
		HistogramF		signal;
		bool			useSplines;
	
  COND_SERIALIZABLE;
};

	std::vector<SigBkg>		pdfs;
	std::vector<double>		bias;
	int				categoryIdx;
	bool				logOutput;
	bool				individual;
	bool				neverUndefined;
	bool				keepEmpty;

  COND_SERIALIZABLE;
};

class ProcLinear : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	std::vector<double>		coeffs;
	double				offset;

  COND_SERIALIZABLE;
};

class ProcMultiply : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	typedef std::vector<unsigned int>	Config;

	unsigned int			in;
	std::vector<Config>		out;

  COND_SERIALIZABLE;
};

class ProcMatrix : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	Matrix				matrix;

  COND_SERIALIZABLE;
};

class ProcExternal : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	virtual std::string getInstanceName() const;

	std::string			method;
	std::vector<unsigned char>	store;

  COND_SERIALIZABLE;
};

class ProcMLP : public VarProcessor {
    public:
#ifndef __GCCXML__
        virtual std::unique_ptr<VarProcessor> clone() const;
#endif
	typedef std::pair<double, std::vector<double> >	Neuron;
	typedef std::pair<std::vector<Neuron>, bool>	Layer;

	std::vector<Layer>		layers;

  COND_SERIALIZABLE;
};

// the discriminator computer

class MVAComputer {
    public:
	MVAComputer();
	MVAComputer(const MVAComputer &orig);
	virtual ~MVAComputer();

	MVAComputer &operator = (const MVAComputer &orig);

	virtual std::vector<VarProcessor*> getProcessors() const;
	void addProcessor(const VarProcessor *proc);

	// cacheId stuff to detect changes
	typedef unsigned int CacheId;
	inline CacheId getCacheId() const { return cacheId; }
	inline bool changed(CacheId old) const { return old != cacheId; }

	std::vector<Variable>		inputSet;
	unsigned int			output;

    private:
	std::vector<VarProcessor*>	processors;

	CacheId				cacheId COND_TRANSIENT;	// transient

  COND_SERIALIZABLE;
};

// a collection of computers identified by name

class MVAComputerContainer {
    public:
	typedef std::pair<std::string, MVAComputer> Entry;

	MVAComputerContainer();
	virtual ~MVAComputerContainer() {}

	MVAComputer &add(const std::string &label);
	virtual const MVAComputer &find(const std::string &label) const;
	virtual bool contains(const std::string &label) const;

	// cacheId stuff to detect changes
	typedef unsigned int CacheId;
	inline CacheId getCacheId() const { return cacheId; }
	inline bool changed(CacheId old) const { return old != cacheId; }

    private:
	std::vector<Entry>	entries;

	CacheId			cacheId COND_TRANSIENT;	// transient

  COND_SERIALIZABLE;
};

} // namespace Calibration
} // namespace PhysicsTools

#endif // CondFormats_PhysicsToolsObjects_MVAComputer_h
