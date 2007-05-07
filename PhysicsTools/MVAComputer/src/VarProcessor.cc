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
// $Id$
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

void VarProcessor::configure(Config_t &config)
{
	if (config.size() != inputVars.size())
		return;

	ConfIterator iter(inputVars.iter(), config);
	configure(iter, nInputVars);
}

void VarProcessor::eval(double *values, int *conf,
                        double *output, int *outConf) const
{
	outConf[1] = outConf[0];

	ValueIterator iter(inputVars.iter(), values, conf, output, outConf);
	eval(iter, nInputVars);
}

} // namespace PhysicsTools
