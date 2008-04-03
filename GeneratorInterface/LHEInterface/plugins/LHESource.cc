#include <iostream>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

#include "LHESource.h"

using namespace lhef;

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	GeneratedInputSource(params, desc),
	reader(new LHEReader(params))
{
	init(params);
}

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc,
                     LHEReader *reader) :
	GeneratedInputSource(params, desc),
	reader(reader)
{
	init(params);
}

void LHESource::init(const edm::ParameterSet &params)
{
	skipEvents = params.getUntrackedParameter<unsigned int>("skipEvents", 0);
	eventsToPrint = params.getUntrackedParameter<unsigned int>("eventsToPrint", 0);
	hadronisation = Hadronisation::create(
		params.getParameter<edm::ParameterSet>("hadronisation"));
	extCrossSect = params.getUntrackedParameter<double>("crossSection", -1.0);
	extFilterEff = params.getUntrackedParameter<double>("filterEfficiency", -1.0);

	if (params.exists("jetMatching")) {
		edm::ParameterSet jetParams =
			params.getUntrackedParameter<edm::ParameterSet>(
								"jetMatching");
		jetMatching = JetMatching::create(jetParams);
	}

	produces<edm::HepMCProduct>();
	produces<edm::GenInfoProduct, edm::InRun>();
	if (jetMatching.get()) {
		produces< std::vector<double> >("matchDeltaR");
		produces< std::vector<double> >("matchDeltaPRel");
	}
}

LHESource::~LHESource()
{
}

void LHESource::endJob()
{
	hadronisation.reset();
	reader.reset();
}

void LHESource::endRun(edm::Run &run)
{
	LHECommon::XSec crossSection;
	if (common)
		crossSection = common->xsec();

	std::auto_ptr<edm::GenInfoProduct> genInfoProd(new edm::GenInfoProduct);

	genInfoProd->set_cross_section(crossSection.value);
	genInfoProd->set_external_cross_section(extCrossSect);
	genInfoProd->set_filter_efficiency(extFilterEff);

	run.put(genInfoProd);

	common.reset();
}

bool LHESource::produce(edm::Event &event)
{
	std::auto_ptr<HepMC::GenEvent> hadronLevel;

	while(true) {
		boost::shared_ptr<LHEEvent> partonLevel = reader->next();
		if (!partonLevel.get())
			return false;

		if (partonLevel->getCommon() != common)
			common = partonLevel->getCommon();

		hadronisation->setEvent(partonLevel);

		hadronLevel = hadronisation->hadronize();

		if (!hadronLevel.get()) {
			if (!skipEvents)
				partonLevel->count(LHECommon::kTried);
			continue;
		}

		if (skipEvents > 0) {
			skipEvents--;
			continue;
		}

		if (jetMatching.get()) {
			double weight = jetMatching->match(
					partonLevel->asHepMCEvent().get(),
					hadronLevel.get());
			if (weight <= 0.0) {
				edm::LogInfo("Generator|LHEInterface")
					<< "Event got rejected by the"
					   "jet matching." << std::endl;
				partonLevel->count(LHECommon::kSelected);
				continue;
			}
		}

		partonLevel->count(LHECommon::kAccepted);
		break;
	}

	hadronLevel->set_event_number(numberEventsInRun()
	                              - remainingEvents() - 1);

	if (eventsToPrint) {
		eventsToPrint--;
		hadronLevel->print();
	}

	std::auto_ptr<edm::HepMCProduct> result(new edm::HepMCProduct);
	result->addHepMCData(hadronLevel.release());
	event.put(result);

	if (jetMatching.get()) {
		std::auto_ptr< std::vector<double> > matchDeltaR(
						new std::vector<double>);
		std::auto_ptr< std::vector<double> > matchDeltaPRel(
						new std::vector<double>);

		typedef std::vector<JetMatching::JetPartonMatch> Matches;
		Matches matches = jetMatching->getMatchSummary();

		for(Matches::const_iterator iter = matches.begin();
		    iter != matches.end(); ++iter) {
			if (!iter->isMatch())
				continue;

			matchDeltaR->push_back(iter->delta);
			matchDeltaPRel->push_back(iter->jet.mag() /
			                          iter->parton.mag() - 1.0);
		}

		event.put(matchDeltaR, "matchDeltaR");
		event.put(matchDeltaPRel, "matchDeltaPRel");
	}

	return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(LHESource);
