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

	virtual void configure(ConfIterator iter, unsigned int n) override;
	virtual ConfigCtx::Context *
	configureLoop(ConfigCtx::Context *ctx, ConfigCtx::iterator begin,
	              ConfigCtx::iterator cur, ConfigCtx::iterator end) override;

	virtual void eval(ValueIterator iter, unsigned int n) const override;
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const override;
	virtual LoopStatus loop(double *output, int *conf,
	                        unsigned int nOutput,
                                LoopCtx& ctx,
	                        unsigned int &nOffset) const override;

    private:
	struct ConfContext : public VarProcessor::ConfigCtx::Context {
		ConfContext(unsigned int origin, unsigned int count) :
			origin(origin), count(count) {}
		virtual ~ConfContext() {}

		unsigned int origin;
		unsigned int count;
	};

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
        auto const offset = iter.loopCtx().offset();
	iter(offset);

        auto& loopSize = iter.loopCtx().size();
	while(iter) {
		unsigned int size = iter.size();
		if (!loopSize)
			loopSize = size;

		double value = iter[offset];
		iter(value);
		iter++;
	}
}

std::vector<double> ProcForeach::deriv(
				ValueIterator iter, unsigned int n) const
{
        auto const offset = iter.loopCtx().offset();
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
                  unsigned int nOutput, LoopCtx& ctx, unsigned int &nOffset) const
{
        auto& index = ctx.index();
	bool endIteration = false;
	if (index++ == count) {
		index = 0;
		endIteration = true;
	}
        auto& offset = ctx.offset();
        auto& size = ctx.size();

	if (offset == 0 && !endIteration) {
		for(int cur = *conf + size; nOutput--; cur += size)
			*++conf = cur;
	}

	if (endIteration) {
		if (++offset >= size) {
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
