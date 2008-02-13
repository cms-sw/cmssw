#include <iostream>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

#include "Pythia6Hadronisation.h"

namespace lhef {

Hadronisation::Hadronisation(const edm::ParameterSet &params)
{
}

Hadronisation::~Hadronisation()
{
}

void Hadronisation::setEvent(const boost::shared_ptr<LHEEvent> &event)
{
	bool newCommon = !rawEvent ||
	                 rawEvent->getCommon() != event->getCommon();
	rawEvent = event;
	if (newCommon)
		this->newCommon(event->getCommon());
}

void Hadronisation::clear()
{
}

Hadronisation *Hadronisation::create(const edm::ParameterSet &params)
{
	std::string name = params.getParameter<std::string>("generator");

	if (name == "Pythia6")
		return new Pythia6Hadronisation(params);
	else
		throw cms::Exception("InvalidGenerator")
			<< "Unknown MC generator \"" << name << "\""
			   " specified for hadronisation in LHESource."
			<< std::endl;
}

void Hadronisation::newCommon(const boost::shared_ptr<LHECommon> &common)
{
}

} // namespace lhef
