// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     VarProcessor
// 

// Implementation:
//     Base class for variable processors. Basically only passes calls
//     through to virtual methods in the actual implementation daughter class.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: VarProcessor.cc,v 1.13 2013/05/23 17:02:16 gartung Exp $
//

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/ProcessRegistry.icc"

// #define DEBUG_DERIV

#ifdef DEBUG_DERIV
#include "FWCore/Utilities/interface/TypeDemangler.h"
#endif

typedef edmplugin::PluginFactory<PhysicsTools::VarProcessor::PluginFunctionPrototype> VPPluginFactory;
EDM_REGISTER_PLUGINFACTORY(VPPluginFactory,
                           "PhysicsToolsMVAComputer");

namespace PhysicsTools {

VarProcessor::VarProcessor(const char *name,
                           const Calibration::VarProcessor *calib,
                           const MVAComputer *computer) :
	computer(computer),
	inputVars(Calibration::convert(calib->inputVars)),
	nInputVars(inputVars.bits())
{
}

VarProcessor::~VarProcessor()
{
	inputVars = BitSet(0);
	nInputVars = 0;
}

void VarProcessor::configure(ConfigCtx &config)
{
	ConfigCtx::size_type pos = config.size();
	if (pos != inputVars.size())
		return;

	ConfIterator iter(inputVars.iter(), config);
	configure(iter, nInputVars);

	VarProcessor *loop = config.loop ? config.loop : this;
	ConfigCtx::Context *ctx =
		loop->configureLoop(config.ctx, config.begin(),
		                    config.begin() + pos, config.end());

	if (ctx != config.ctx) {
		delete config.ctx;
		config.ctx = ctx;
	}

	if (config.loop && !ctx)
		config.loop = 0;
	else if (!config.loop && ctx)
		config.loop = this;
}

VarProcessor::ConfigCtx::ConfigCtx(const std::vector<Variable::Flags>& flags) :
	loop(0), ctx(0)
{
	for(std::vector<Variable::Flags>::const_iterator iter = flags.begin();
	    iter != flags.end(); ++iter)
		configs.push_back(Config(*iter, 1));
}

VarProcessor::ConfigCtx::Context *
VarProcessor::configureLoop(ConfigCtx::Context *ctx, ConfigCtx::iterator begin,
                            ConfigCtx::iterator cur, ConfigCtx::iterator end)
{
	return 0;
}

template<>
VarProcessor *ProcessRegistry<VarProcessor, Calibration::VarProcessor,
                              const MVAComputer>::Factory::create(
	        const char *name, const Calibration::VarProcessor *calib,
		const MVAComputer *parent)
{
	VarProcessor *result = ProcessRegistry::create(name, calib, parent);
	if (!result) {
		// try to load the shared library and retry
		try {
			delete VPPluginFactory::get()->create(
					std::string("VarProcessor/") + name);
			result = ProcessRegistry::create(name, calib, parent);
		} catch(const cms::Exception &e) {
			// caller will have to deal with the null pointer
			// in principle this will just give a slightly more
			// descriptive error message (and will rethrow anyhow)

                  edm::LogError("CannotBuildMVAProc")
                    << "Caught exception when building processor: "
                    << name << " message: " << std::endl
                    << e.what() << std::endl;
                  throw e;
		}
	}
	return result;
}

void VarProcessor::deriv(double *input, int *conf, double *output,
                         int *outConf, int *loop, unsigned int offset,
                         unsigned int in, unsigned int out_,
                         std::vector<double> &deriv) const
{
	ValueIterator iter(inputVars.iter(), input, conf,
	                   output, outConf, loop, offset);

	eval(iter, nInputVars);

	std::vector<double> matrix = this->deriv(iter, nInputVars);

	unsigned int size = 0;
	while(iter)
		size += (iter++).size();
        bool empty = matrix.empty();
        assert(size != 0 || empty);
	unsigned int out = empty ? 0 : (matrix.size() / size);

	if (matrix.size() != out * size ||
	    (out > 1 && (int)out != outConf[out_] - outConf[0]))
		throw cms::Exception("VarProcessor")
			<< "Derivative matrix implausible size in "
			<< typeid(*this).name() << "."
			<< std::endl;

#ifdef DEBUG_DERIV
	if (!matrix.empty()) {
                std::string demangledName;
                edm::typeDemangle(typeid(*this).name(), demangledName);
                std::cout << demangledName << std::endl;
		for(unsigned int i = 0; i < out; i++) {
			for(unsigned int j = 0; j < size; j++)
				std::cout << matrix.at(i*size+j) << "\t";
			std::cout << std::endl;
		}
		std::cout << "----------------" << std::endl;
	}

	std::cout << "======= in = " << in << ", size = " << size
	          << ", out = " << out << ", matrix = " << matrix.size()
	          << std::endl;
#endif

	unsigned int sz = (outConf[out_] - in) * in;
	unsigned int oldSz = deriv.size();
	if (oldSz < sz)
		deriv.resize(sz);

	double *begin = &deriv.front() + (outConf[0] - in + offset) * in;
	double *end = begin + out * in;
	if (begin < &deriv.front() + oldSz)
		std::fill(begin, end, 0.0);

	if (matrix.empty())
		return;

	double *m0 = &matrix.front();
	BitSet::Iterator cur = inputVars.iter();
	for(unsigned int i = 0; i < nInputVars; i++, ++cur) {
#ifdef DEBUG_DERIV
		std::cout << " inputvar " << i << std::endl;
#endif
		int *curConf = conf + cur();
		unsigned int pos = *curConf;
#ifdef DEBUG_DERIV
		std::cout << " -> cur = " << cur() << ", pos = "
		          << pos << std::endl;
#endif
		if (loop && curConf >= loop) {
			pos += offset;
			loop = 0;
		}

		unsigned int n = loop ? (curConf[1] - curConf[0]) : 1;
		for(unsigned int j = 0; j < n; m0++, j++, pos++) {
#ifdef DEBUG_DERIV
			std::cout << "  multip " << j << std::endl;
#endif
			double *p = begin;
			if (pos >= in) {
#ifdef DEBUG_DERIV
				std::cout << "   deriv " << (pos - in)
				          << std::endl;
#endif
				const double *q = &deriv.front() +
				                  (pos - in) * in;
				for(const double *m = m0; p < end;
				    m += size, q -= in)
					for(unsigned int k = 0; k < in; k++)
						*p++ += *m * *q++;
			} else {
#ifdef DEBUG_DERIV
				std::cout << "   in " << pos << std::endl;
#endif
				for(const double *m = m0; p < end;
				    m += size, p += in)
					p[pos] += *m;
			}
		}
	}

#ifdef DEBUG_DERIV
	std::cout << "================" << std::endl;
	for(const double *p = &deriv.front();
	    p != &deriv.front() + deriv.size();) {
		for(unsigned int j = 0; j < in; j++)
			std::cout << *p++ << "\t";
		std::cout << std::endl;
	}
	std::cout << "================" << std::endl;
#endif
}

} // namespace PhysicsTools

// Force instantiation of its templated static methods.
template void PhysicsTools::ProcessRegistry<PhysicsTools::VarProcessor, PhysicsTools::Calibration::VarProcessor, PhysicsTools::MVAComputer const>::unregisterProcess(char const*);
template void PhysicsTools::ProcessRegistry<PhysicsTools::VarProcessor, PhysicsTools::Calibration::VarProcessor, PhysicsTools::MVAComputer const>::registerProcess(char const*, PhysicsTools::ProcessRegistry<PhysicsTools::VarProcessor, PhysicsTools::Calibration::VarProcessor, PhysicsTools::MVAComputer const> const*);
