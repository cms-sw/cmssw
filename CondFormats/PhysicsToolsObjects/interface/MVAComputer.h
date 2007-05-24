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
// $Id: MVAComputer.h,v 1.7 2007/05/21 02:00:22 saout Exp $
//

#include <string>
#include <vector>
#include <map>

namespace PhysicsTools {
namespace Calibration {

// forward declarations

typedef std::pair<double, double> MinMax;

// helper classes

class BitSet {
    public:
	std::vector<unsigned char>	store;
	unsigned int			bitsInLast;
};

class Matrix {
    public:
	std::vector<double>		elements;
	unsigned int			rows;
	unsigned int			columns;
};

class PDF {
    public:
	std::vector<double>		distr;
	MinMax				range;
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

class ProcNormalize : public VarProcessor {
    public:
	std::vector<PDF>		distr;
};

class ProcLikelihood : public VarProcessor {
    public:
	class SigBkg {
	    public:
		PDF			signal;
		PDF			background;
	};

	std::vector<SigBkg>		pdfs;
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
	std::vector<ProcNormalize>	vProcNormalize_;
	std::vector<ProcLikelihood>	vProcLikelihood_;
	std::vector<ProcLinear>		vProcLinear_;
	std::vector<ProcMultiply>	vProcMultiply_;
	std::vector<ProcMatrix>		vProcMatrix_;
	std::vector<ProcTMVA>		vProcTMVA_;
	std::vector<ProcMLP>		vProcMLP_;

	CacheId				cacheId;	// transient
};

// this is a temporary hack used in RecoBTau until ESSources can be
// retrieved via label from the same record
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
