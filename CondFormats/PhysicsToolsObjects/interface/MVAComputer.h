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
// $Id: MVAComputer.h,v 1.13 2007/12/08 15:57:07 saout Exp $
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
	// help that poor ROOT Cint/Reflex to copy bitsets... (workaround)
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
	std::vector<HistogramF>	distr;
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

	enum Flags {
		kCategoryMax	= 19,
		kLogOutput,
		kIndividual,
		kNeverUndefined,
		kKeepEmpty
	};

	std::vector<SigBkg>		pdfs;
	std::vector<double>		bias;
	int				categoryIdx;
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

class ProcTMVA : public VarProcessor {
    public:
	std::string			method;
	std::vector<std::string>	variables;
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
	virtual ~MVAComputer();

	std::vector<Variable>		inputSet;
//	std::vector<VarProcessor*>	processors;	// stupid POOL
	virtual std::vector<VarProcessor*> getProcessors() const;
	void				addProcessor(const VarProcessor *proc);
	unsigned int			output;

	// cacheId stuff to detect changes
	typedef unsigned int CacheId;
	inline CacheId getCacheId() const { return cacheId; }
	inline bool changed(CacheId old) const { return old != cacheId; }

	// these variables are read/written only via get/setProcessor()
	// ordering is relevant for the persistent storage
    private:
	std::vector<unsigned int>	processors_;

	std::vector<ProcOptional>	vProcOptional_;
	std::vector<ProcCount>		vProcCount_;
	std::vector<ProcClassed>	vProcClassed_;
	std::vector<ProcSplitter>	vProcSplitter_;
	std::vector<ProcForeach>	vProcForeach_;
	std::vector<ProcSort>		vProcSort_;
	std::vector<ProcCategory>	vProcCategory_;
	std::vector<ProcNormalize>	vProcNormalize_;
	std::vector<ProcLikelihood>	vProcLikelihood_;
	std::vector<ProcLinear>		vProcLinear_;
	std::vector<ProcMultiply>	vProcMultiply_;
	std::vector<ProcMatrix>		vProcMatrix_;
	std::vector<ProcTMVA>		vProcTMVA_;
	std::vector<ProcMLP>		vProcMLP_;

	CacheId				cacheId;	// transient
};

// useful if different categories exist with different configurations
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
