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

namespace {
	class NoHadronisation : public lhef::Hadronisation {
	    public:
		NoHadronisation(const edm::ParameterSet &params) :
			Hadronisation(params) {}
		~NoHadronisation() {}

	    private:
		std::auto_ptr<HepMC::GenEvent> doHadronisation()
		{ return getRawEvent()->asHepMCEvent(); }
};

} // anonymous namespae

namespace lhef {

Hadronisation::Hadronisation(const edm::ParameterSet &params)
{
}

Hadronisation::~Hadronisation()
{
}

bool Hadronisation::setEvent(const boost::shared_ptr<LHEEvent> &event)
{
	bool newCommon = !rawEvent ||
	                 (rawEvent->getCommon() != event->getCommon() &&
	                  *rawEvent->getCommon() != *event->getCommon());
	rawEvent = event;
	if (newCommon) {
		this->newCommon(event->getCommon());
		return true;
	} else
		return false;
}

void Hadronisation::clear()
{
}

double Hadronisation::getCrossSection() const
{
	return -1.0;
}

std::auto_ptr<Hadronisation> Hadronisation::create(
					const edm::ParameterSet &params)
{
	std::string name = params.getParameter<std::string>("generator");

	if (name == "None")
		return std::auto_ptr<Hadronisation>(
					new NoHadronisation(params));

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

std::auto_ptr<HepMC::GenEvent> Hadronisation::hadronize()
{
	std::auto_ptr<HepMC::GenEvent> event = this->doHadronisation();

	if (!event.get())
		return event;

	const HepMC::GenVertex *signalVertex = event->signal_process_vertex();
	if (!signalVertex) {
		signalVertex = findSignalVertex(event.get());
		event->set_signal_process_vertex(
			const_cast<HepMC::GenVertex*>(signalVertex));
	}

	return event;
}

void Hadronisation::newCommon(const boost::shared_ptr<LHECommon> &common)
{
}

const HepMC::GenVertex *Hadronisation::findSignalVertex(
						const HepMC::GenEvent *event)
{
	double largestMass2 = -9.0e-30;
	const HepMC::GenVertex *vertex = 0;
	for(HepMC::GenEvent::vertex_const_iterator iter = event->vertices_begin();
	    iter != event->vertices_end(); ++iter) {
		double px = 0.0, py = 0.0, pz = 0.0, E = 0.0;
		bool hadStatus3 = false;
		for(HepMC::GenVertex::particles_out_const_iterator iter2 =
					(*iter)->particles_out_const_begin();
		    iter2 != (*iter)->particles_out_const_end(); ++iter2) {
			hadStatus3 = hadStatus3 || (*iter2)->status() == 3;
			px += (*iter2)->momentum().px();
			py += (*iter2)->momentum().py();
			pz += (*iter2)->momentum().pz();
			E += (*iter2)->momentum().e();
		}
		if (!hadStatus3)
			continue;

		double mass2 = E * E - (px * px + py * py + pz * pz);
		if (mass2 > largestMass2) {
			vertex = *iter;
			largestMass2 = mass2;
		}
	}

	return vertex;
}

} // namespace lhef
