#include <iostream>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

EDM_REGISTER_PLUGINFACTORY(lhef::Hadronisation::Factory,
                           "GeneratorInterfaceLHEHadronisation");

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

std::auto_ptr<Hadronisation> Hadronisation::create(
					const edm::ParameterSet &params)
{
	std::string name = params.getParameter<std::string>("generator");

	std::auto_ptr<Hadronisation> plugin(
		Factory::get()->create(name + "Hadronisation", params));

	if (!plugin.get())
		throw cms::Exception("InvalidGenerator")
			<< "Unknown MC generator \"" << name << "\""
			   " specified for hadronisation in LHESource."
			<< std::endl;

	edm::LogInfo("Generator|LHEInterface")
		<< "Using name to hadronize LHE input." << std::endl;

	return plugin;
}

void Hadronisation::newCommon(const boost::shared_ptr<LHECommon> &common)
{
}

} // namespace lhef
