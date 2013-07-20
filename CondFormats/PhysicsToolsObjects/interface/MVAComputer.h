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
// $Id: MVAComputer.h,v 1.16 2012/08/23 17:59:29 wmtan Exp $
//

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
};

class Matrix {
    public:
	std::vector<double>		elements;
	unsigned int			rows;
	unsigned int			columns;
};

// configuration base classes

class VarProcessor {
    public:
	BitSet	inputVars;

	virtual ~VarProcessor() {}
	virtual std::string getInstanceName() const;
};

class Variable {
    public:
	inline Variable() {}
	inline Variable(const std::string &name) : name(name) {}
	inline ~Variable() {}

	std::string			name;
};

// variable processors

class ProcOptional : public VarProcessor {
    public:
	std::vector<double>		neutralPos;
};

class ProcCount : public VarProcessor {};

class ProcClassed : public VarProcessor {
    public:
	unsigned int			nClasses;
};

class ProcSplitter : public VarProcessor {
    public:
	unsigned int			nFirst;
};

class ProcForeach : public VarProcessor {
    public:
	unsigned int			nProcs;
};

class ProcSort : public VarProcessor {
    public:
	unsigned int			sortByIndex;
	bool				descending;
};

class ProcCategory : public VarProcessor {
    public:
	typedef std::vector<double> BinLimits;

	std::vector<BinLimits>		variableBinLimits;
	std::vector<int>		categoryMapping;
};

class ProcNormalize : public VarProcessor {
    public:
	std::vector<HistogramF>		distr;
	int				categoryIdx;
};

class ProcLikelihood : public VarProcessor {
    public:
	class SigBkg {
	    public:
		HistogramF		background;
		HistogramF		signal;
		bool			useSplines;
	};

	std::vector<SigBkg>		pdfs;
	std::vector<double>		bias;
	int				categoryIdx;
	bool				logOutput;
	bool				individual;
	bool				neverUndefined;
	bool				keepEmpty;
};

class ProcLinear : public VarProcessor {
    public:
	std::vector<double>		coeffs;
	double				offset;
};

class ProcMultiply : public VarProcessor {
    public:
	typedef std::vector<unsigned int>	Config;

	unsigned int			in;
	std::vector<Config>		out;
};

class ProcMatrix : public VarProcessor {
    public:
	Matrix				matrix;
};

class ProcExternal : public VarProcessor {
    public:
	virtual std::string getInstanceName() const;

	std::string			method;
	std::vector<unsigned char>	store;
};

class ProcMLP : public VarProcessor {
    public:
	typedef std::pair<double, std::vector<double> >	Neuron;
	typedef std::pair<std::vector<Neuron>, bool>	Layer;

	std::vector<Layer>		layers;
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

	CacheId				cacheId;	// transient
};

// a collection of computers identified by name

class MVAComputerContainer {
    public:
	typedef std::pair<std::string, MVAComputer> Entry;

	MVAComputerContainer();
	virtual ~MVAComputerContainer() {}

	MVAComputer &add(const std::string &label);
	virtual const MVAComputer &find(const std::string &label) const;

	// cacheId stuff to detect changes
	typedef unsigned int CacheId;
	inline CacheId getCacheId() const { return cacheId; }
	inline bool changed(CacheId old) const { return old != cacheId; }

    private:
	std::vector<Entry>	entries;

	CacheId			cacheId;	// transient
};

} // namespace Calibration
} // namespace PhysicsTools

#endif // CondFormats_PhysicsToolsObjects_MVAComputer_h
