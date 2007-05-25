// -*- C++ -*-
//
// Package:     Discriminator
// Class  :     VarProcessor
// 

// Implementation:
//     Base class for variable processors. Basically only passes calls
//     through to virtual methods in the actual implementation daughter class.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: VarProcessor.cc,v 1.1 2007/05/07 18:30:55 saout Exp $
//

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/BitSet.h"

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

VarProcessor::ConfigCtx::Context *
VarProcessor::configureLoop(ConfigCtx::Context *ctx, ConfigCtx::iterator begin,
                            ConfigCtx::iterator cur, ConfigCtx::iterator end)
{
	return 0;
}

} // namespace PhysicsTools
