#include <iostream>
#include <vector>
#include <string>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/Processor.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcCount : public Processor {
    public:
	typedef Processor::Registry<ProcCount>::Type Registry;

	ProcCount(const char *name, const AtomicId *id,
	             MVATrainer *trainer);
	virtual ~ProcCount();

	virtual Variable::Flags getDefaultFlags() const
	{ return Variable::FLAG_ALL; }

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalib() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values, bool target);
	virtual void trainEnd();

    private:
	std::vector<double>	neutrals;
};

static ProcCount::Registry registry("ProcCount");

ProcCount::ProcCount(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	Processor(name, id, trainer)
{
}

ProcCount::~ProcCount()
{
}

void ProcCount::configure(DOMElement *elem)
{
	trained = true;
}

Calibration::VarProcessor *ProcCount::getCalib() const
{
	return new Calibration::ProcCount;
}

void ProcCount::trainBegin()
{
}

void ProcCount::trainData(const std::vector<double> *values, bool target)
{
}

void ProcCount::trainEnd()
{
}

} // anonymous namespace
