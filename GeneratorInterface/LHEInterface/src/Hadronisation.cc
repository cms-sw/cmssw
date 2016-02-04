#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
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
		void doInit() {}
		std::auto_ptr<HepMC::GenEvent> doHadronisation()
		{ return getRawEvent()->asHepMCEvent(); }
};

} // anonymous namespae

namespace lhef {

Hadronisation::Hadronisation(const edm::ParameterSet &params) :
	psRequested(false),
	psAsHepMC(true)
{
}

Hadronisation::~Hadronisation()
{
}

void Hadronisation::init()
{
	doInit();
}

bool Hadronisation::setEvent(const boost::shared_ptr<LHEEvent> &event)
{
	bool newRunInfo = !rawEvent ||
	                  (rawEvent->getRunInfo() != event->getRunInfo() &&
	                   *rawEvent->getRunInfo() != *event->getRunInfo());
	rawEvent = event;
	if (newRunInfo) {
		this->newRunInfo(event->getRunInfo());
		return true;
	} else
		return false;
}

void Hadronisation::clear()
{
}

std::set<std::string> Hadronisation::capabilities() const
{
	return std::set<std::string>();
}

void Hadronisation::matchingCapabilities(
				const std::set<std::string> &capabilities)
{
	psRequested = false;
	psAsHepMC = false;
	for(std::set<std::string>::const_iterator iter = capabilities.begin();
	    iter != capabilities.end(); ++iter) {
		if (*iter == "hepmc")
			psAsHepMC = true;
		else if (*iter == "psFinalState")
			psRequested = true;
		else if (*iter == "matchSummary")
			/* nothing */;
		else if (!this->capabilities().count(*iter))
			throw cms::Exception("Generator|LHEInterface")
				<< "JetMatching expected capability \""
				<< *iter << "\", but hadronizer does not "
				   "support it." << std::endl;
	}
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
			   " specified for hadronisation in LHEProducer."
			<< std::endl;

	edm::LogInfo("Generator|LHEInterface")
		<< "Using " << name << " to hadronize LHE input." << std::endl;

	return plugin;
}

std::auto_ptr<HepMC::GenEvent> Hadronisation::hadronize()
{
	std::auto_ptr<HepMC::GenEvent> event = this->doHadronisation();
	if (!event.get())
		return event;

	const HepMC::GenVertex *signalVertex = event->signal_process_vertex();
	if (!signalVertex) {
		signalVertex = LHEEvent::findSignalVertex(event.get());
		event->set_signal_process_vertex(
			const_cast<HepMC::GenVertex*>(signalVertex));
	}

	return event;
}

void Hadronisation::newRunInfo(const boost::shared_ptr<LHERunInfo> &runInfo)
{
}

bool Hadronisation::showeredEvent(
			const boost::shared_ptr<HepMC::GenEvent> &event)
{
	if (event.get()) {
		const HepMC::GenVertex *signalVertex =
					event->signal_process_vertex();
		if (!signalVertex) {
			signalVertex = LHEEvent::findSignalVertex(event.get(), false);
			event->set_signal_process_vertex(
				const_cast<HepMC::GenVertex*>(signalVertex));
		}
	}

	return sigShower.emit(event);
}

} // namespace lhef
