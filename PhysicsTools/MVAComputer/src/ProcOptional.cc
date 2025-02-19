// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     ProcOptional
// 

// Implementation:
//     Variable processor to set empty input variables to a default value.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: ProcOptional.cc,v 1.4 2009/06/03 09:50:14 saout Exp $
//

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"

using namespace PhysicsTools;

namespace { // anonymous

class ProcOptional : public VarProcessor {
    public:
	typedef VarProcessor::Registry::Registry<ProcOptional,
					Calibration::ProcOptional> Registry;

	ProcOptional(const char *name,
	             const Calibration::ProcOptional *calib,
	             const MVAComputer *computer);
	virtual ~ProcOptional() {}

	virtual void configure(ConfIterator iter, unsigned int n);
	virtual void eval(ValueIterator iter, unsigned int n) const;
	virtual std::vector<double> deriv(
				ValueIterator iter, unsigned int n) const;

    private:
	std::vector<double>	neutralPos;
};

static ProcOptional::Registry registry("ProcOptional");

ProcOptional::ProcOptional(const char *name,
                          const Calibration::ProcOptional *calib,
                          const MVAComputer *computer) :
	VarProcessor(name, calib, computer),
	neutralPos(calib->neutralPos)
{
}

void ProcOptional::configure(ConfIterator iter, unsigned int n)
{
	if (n != neutralPos.size())
		return;

	while(iter)
		iter++(Variable::FLAG_OPTIONAL) << Variable::FLAG_NONE;
}

void ProcOptional::eval(ValueIterator iter, unsigned int n) const
{
	for(std::vector<double>::const_iterator pos = neutralPos.begin();
	    pos != neutralPos.end(); pos++, ++iter) {
		switch(iter.size()) {
		    case 0:
			iter(*pos);
			break;
		    case 1:
			iter(*iter);
			break;
		    default:
			throw cms::Exception("ProcOptional")
				<< "Multiple input variables encountered."
				<< std::endl;
		}
	}
}

std::vector<double> ProcOptional::deriv(
				ValueIterator iter, unsigned int n) const
{
	unsigned int size = 0;
	for(ValueIterator iter2 = iter; iter2; ++iter2)
		size += iter2.size();

	std::vector<double> result;

	unsigned int column = 0;
	for(std::vector<double>::const_iterator pos = neutralPos.begin();
	    pos != neutralPos.end(); pos++, ++iter) {
		unsigned int row = result.size();
		result.resize(row + size);
		switch(iter.size()) {
		    case 0:
			break;
		    case 1:
			result[row + column++] = 1.0;
			break;
		    default:
			throw cms::Exception("ProcOptionalError")
				<< "Multiple input variables encountered."
				<< std::endl;
		}
	}

	return result;
}

} // anonymous namespace
