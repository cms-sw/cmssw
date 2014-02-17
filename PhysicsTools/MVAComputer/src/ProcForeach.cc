// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcForeach
// 

// Implementation:
//     Loops over a specified amount of VarProcessors and passes each
//     instance of a set of variables individually per iteration.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcForeach.cc,v 1.3 2009/06/03 09:50:14 saout Exp $
//

#include <algorithm>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcForeach : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcForeach,
					Calibration::ProcForeach> Registry;

	ProcForeach(const char *name,
	             const Calibration::ProcForeach *calib,
	             const MVAComputer *computer);
	virtual ~ProcForeach() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual ConfigCtx::Context *
	configureLoop(ConfigCtx::Context *ctx, ConfigCtx::iterator begin,
	              ConfigCtx::iterator cur, ConfigCtx::iterator end);

	virtual void eval(ValueIterator iter, unsigned int n) const;
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const;
	virtual LoopStatus loop(double *output, int *conf,
	                        unsigned int nOutput,
	                        unsigned int &nOffset) const;

    private:
	struct ConfContext : public VarProcessor::ConfigCtx::Context {
		ConfContext(unsigned int origin, unsigned int count) :
			origin(origin), count(count) {}
		virtual ~ConfContext() {}

		unsigned int origin;
		unsigned int count;
	};

	inline void reset() const { index = offset = size = 0; }

	mutable unsigned int	index;
	mutable unsigned int	offset;
	mutable unsigned int	size;

	unsigned int		count;
};

static ProcForeach::Registry registry("ProcForeach");

ProcForeach::ProcForeach(const char *name,
                         const Calibration::ProcForeach *calib,
                         const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	count(calib->nProcs)
{
}

void ProcForeach::configure(ConfIterator iter, unsigned int n)
{
	reset();
	iter << Variable::FLAG_NONE;
	while(iter)
		iter << iter++(Variable::FLAG_MULTIPLE);
}

VarProcessor::ConfigCtx::Context *
ProcForeach::configureLoop(ConfigCtx::Context *ctx_, ConfigCtx::iterator begin,
                           ConfigCtx::iterator cur, ConfigCtx::iterator end)
{
	ConfContext *ctx = dynamic_cast<ConfContext*>(ctx_);
	if (!ctx)
		return new ConfContext(cur - begin + 1, count);

	for(ConfigCtx::iterator iter = cur; iter != end; iter++) {
		iter->mask = Variable::FLAG_ALL;
		iter->origin = ctx->origin;
	}

	if (--ctx->count)
		return ctx;
	else
		return 0;
}

void ProcForeach::eval(ValueIterator iter, unsigned int n) const
{
	iter(offset);

	while(iter) {
		unsigned int size = iter.size();
		if (!this->size)
			this->size = size;

		double value = iter[offset];
		iter(value);
		iter++;
	}
}

std::vector<double> ProcForeach::deriv(
				ValueIterator iter, unsigned int n) const
{
	std::vector<unsigned int> offsets;
	unsigned int in = 0, out = 0;
	while(iter) {
		offsets.push_back(in + offset);
		in += (iter++).size();
		out++;
	}

	std::vector<double> result((out + 1) * in, 0.0);
	for(unsigned int i = 0; i < out; i++)
		result[(i + 1) * in + offsets[i]] = 1.0;

	return result;
}

VarProcessor::LoopStatus
ProcForeach::loop(double *output, int *conf,
                  unsigned int nOutput, unsigned int &nOffset) const
{
	bool endIteration = false;
	if (index++ == count) {
		index = 0;
		endIteration = true;
	}

	if (offset == 0 && !endIteration) {
		for(int cur = *conf + size; nOutput--; cur += size)
			*++conf = cur;
	}

	if (endIteration) {
		if (++offset >= size) {
			reset();
			return kStop;
		} else
			return kReset;
	} else if (offset > size) {
		return kSkip;
	} else {
		nOffset = offset;
		return kNext;
	}
}

} // anonymous namespace
