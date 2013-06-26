#include <iostream>
#include <vector>
#include <string>

#include <xercesc/dom/DOM.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcCount : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcCount>::Type Registry;

	ProcCount(const char *name, const AtomicId *id,
	          MVATrainer *trainer);
	virtual ~ProcCount();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

    private:
	std::vector<double>	neutrals;
};

static ProcCount::Registry registry("ProcCount");

ProcCount::ProcCount(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	TrainProcessor(name, id, trainer)
{
}

ProcCount::~ProcCount()
{
}

void ProcCount::configure(DOMElement *elem)
{
	trained = true;
}

Calibration::VarProcessor *ProcCount::getCalibration() const
{
	return new Calibration::ProcCount;
}

} // anonymous namespace
