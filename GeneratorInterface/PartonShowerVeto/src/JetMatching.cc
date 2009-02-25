#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"


namespace gen {

JetMatching::JetMatching(const edm::ParameterSet &params)
{
   fMatchingStatus=false;
}

JetMatching::~JetMatching()
{
}

void JetMatching::init(const lhef::LHERunInfo* runInfo)
{
}

void JetMatching::beforeHadronisation(const lhef::LHEEvent* event)
{
}

void JetMatching::beforeHadronisationExec()
{
}

std::set<std::string> JetMatching::capabilities() const
{
	std::set<std::string> result;
	result.insert("psFinalState");
	result.insert("hepmc");
	return result;
}

} // namespace gen
