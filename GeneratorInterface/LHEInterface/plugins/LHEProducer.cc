#include <iostream>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

using namespace lhef;

class LHEProducer : public edm::EDProducer {
    public:
	explicit LHEProducer(const edm::ParameterSet &params);
	virtual ~LHEProducer();

    protected:
	virtual void beginJob();
	virtual void endJob();
	virtual void beginRun(edm::Run &run, const edm::EventSetup &es);
	virtual void endRun(edm::Run &run, const edm::EventSetup &es);
	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	unsigned int				eventsToPrint;
	std::auto_ptr<lhef::Hadronisation>	hadronisation;
	std::auto_ptr<lhef::JetMatching>	jetMatching;

	double					extCrossSect;
	double					extFilterEff;

	boost::shared_ptr<lhef::LHECommon>	common;
	unsigned int				index;
};

LHEProducer::LHEProducer(const edm::ParameterSet &params) :
	eventsToPrint(params.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	extCrossSect(params.getUntrackedParameter<double>("crossSection", -1.0)),
	extFilterEff(params.getUntrackedParameter<double>("filterEfficiency", -1.0))
{
	hadronisation = Hadronisation::create(
		params.getParameter<edm::ParameterSet>("hadronisation"));

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

LHEProducer::~LHEProducer()
{
}

void LHEProducer::beginJob()
{
}

void LHEProducer::endJob()
{
	hadronisation.reset();
}

void LHEProducer::beginRun(edm::Run &run, const edm::EventSetup &es)
{
	edm::Handle<HEPRUP> heprup;
	run.getByLabel("source", heprup);

	common.reset(new LHECommon(*heprup, ""));
	index = 0;
}

void LHEProducer::endRun(edm::Run &run, const edm::EventSetup &es)
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

void LHEProducer::produce(edm::Event &event, const edm::EventSetup &es)
{
	std::auto_ptr<HepMC::GenEvent> hadronLevel;

	edm::Handle<HEPEUP> hepeup;
	event.getByLabel("source", hepeup);

	boost::shared_ptr<LHEEvent> partonLevel(new LHEEvent(common, *hepeup));

	hadronisation->setEvent(partonLevel);

	hadronLevel = hadronisation->hadronize();

	if (!hadronLevel.get())
		return;

	partonLevel->count(LHECommon::kTried);

	if (jetMatching.get()) {
		double weight = jetMatching->match(
					partonLevel->asHepMCEvent().get(),
					hadronLevel.get());
		if (weight <= 0.0) {
			edm::LogInfo("Generator|LHEInterface")
				<< "Event got rejected by the"
				   "jet matching." << std::endl;
			partonLevel->count(LHECommon::kSelected);
			return;
		}
	}

	partonLevel->count(LHECommon::kAccepted);

	hadronLevel->set_event_number(++index);

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
}

DEFINE_ANOTHER_FWK_MODULE(LHEProducer);
