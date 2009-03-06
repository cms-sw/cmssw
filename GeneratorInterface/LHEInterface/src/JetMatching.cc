#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

EDM_REGISTER_PLUGINFACTORY(lhef::JetMatching::Factory,
                           "GeneratorInterfaceLHEJetMatching");

namespace lhef {

JetMatching::JetMatching(const edm::ParameterSet &params)
{
}

JetMatching::~JetMatching()
{
}

void JetMatching::init(const boost::shared_ptr<LHERunInfo> &runInfo)
{
}

void JetMatching::beforeHadronisation(
				const boost::shared_ptr<LHEEvent> &event)
{
}

std::set<std::string> JetMatching::capabilities() const
{
	std::set<std::string> result;
	result.insert("psFinalState");
	result.insert("hepmc");
	return result;
}

std::auto_ptr<JetMatching> JetMatching::create(
					const edm::ParameterSet &params)
{
	std::string name = params.getParameter<std::string>("scheme");

	std::auto_ptr<JetMatching> plugin(
		Factory::get()->create("JetMatching" + name, params));

	if (!plugin.get())
		throw cms::Exception("InvalidJetMatching")
			<< "Unknown scheme \"" << name << "\""
			   " specified for jet matching in LHEProducer."
			<< std::endl;

	edm::LogInfo("Generator|LHEInterface")
		<< "Using " << name << " for ME/PS jet matching." << std::endl;

	return plugin;
}

} // namespace lhef
