#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>
#include <sigc++/signal.h>

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

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

using namespace lhef;

class LHEProducer : public edm::EDProducer {
    public:
	explicit LHEProducer(const edm::ParameterSet &params);
	virtual ~LHEProducer();

    protected:
	virtual void beginJob(const edm::EventSetup &es);
	virtual void endJob();
	virtual void beginRun(edm::Run &run, const edm::EventSetup &es);
	virtual void endRun(edm::Run &run, const edm::EventSetup &es);
	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	double matching(const HepMC::GenEvent *event, bool shower) const;

	bool showeredEvent(const boost::shared_ptr<HepMC::GenEvent> &event);

	unsigned int			eventsToPrint;
	std::vector<int>		removeResonances;
	std::auto_ptr<Hadronisation>	hadronisation;
	std::auto_ptr<JetMatching>	jetMatching;

	double				extCrossSect;
	double				extFilterEff;

	boost::shared_ptr<LHEEvent>	partonLevel;
	boost::shared_ptr<LHERunInfo>	runInfo;
	unsigned int			index;
	bool				matchingDone;
	double				weight;
};

LHEProducer::LHEProducer(const edm::ParameterSet &params) :
	eventsToPrint(params.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	extCrossSect(params.getUntrackedParameter<double>("crossSection", -1.0)),
	extFilterEff(params.getUntrackedParameter<double>("filterEfficiency", -1.0))
{
	hadronisation = Hadronisation::create(
		params.getParameter<edm::ParameterSet>("hadronisation"));

	if (params.exists("removeResonances"))
		removeResonances =
			params.getParameter<std::vector<int> >(
							"removeResonances");

	if (params.exists("jetMatching")) {
		edm::ParameterSet jetParams =
			params.getUntrackedParameter<edm::ParameterSet>(
								"jetMatching");
		jetMatching = JetMatching::create(jetParams);
	}

	produces<edm::HepMCProduct>();
	produces<edm::GenInfoProduct, edm::InRun>();

	if (jetMatching.get()) {
		if (params.getUntrackedParameter<bool>(
					"preferShowerVetoCallback", true))
			hadronisation->onShoweredEvent().connect(
				sigc::mem_fun(*this,
				              &LHEProducer::showeredEvent));

		produces< std::vector<double> >("matchDeltaR");
		produces< std::vector<double> >("matchDeltaPRel");
	}
}

LHEProducer::~LHEProducer()
{
}

void LHEProducer::beginJob(const edm::EventSetup &es)
{
	hadronisation->init();
}

void LHEProducer::endJob()
{
	hadronisation.reset();
}

void LHEProducer::beginRun(edm::Run &run, const edm::EventSetup &es)
{
	edm::Handle<LHERunInfoProduct> product;
	run.getByLabel("source", product);

	runInfo.reset(new LHERunInfo(product->heprup()));
	index = 0;
}

void LHEProducer::endRun(edm::Run &run, const edm::EventSetup &es)
{
	LHERunInfo::XSec crossSection;
	if (runInfo)
		crossSection = runInfo->xsec();

	std::auto_ptr<edm::GenInfoProduct> genInfoProd(new edm::GenInfoProduct);

	genInfoProd->set_cross_section(crossSection.value);
	genInfoProd->set_external_cross_section(extCrossSect);
	genInfoProd->set_filter_efficiency(extFilterEff);

	run.put(genInfoProd);

	runInfo.reset();
}

void LHEProducer::produce(edm::Event &event, const edm::EventSetup &es)
{
	std::auto_ptr<edm::HepMCProduct> result(new edm::HepMCProduct);

	edm::Handle<LHEEventProduct> product;
	event.getByLabel("source", product);

	partonLevel.reset(new LHEEvent(runInfo, product->hepeup()));
	if (!removeResonances.empty())
		partonLevel->removeResonances(removeResonances);

	if (product->pdf())
		partonLevel->setPDF(
			std::auto_ptr<LHEEvent::PDF>(
				new LHEEvent::PDF(*product->pdf())));

	hadronisation->setEvent(partonLevel);

	matchingDone = false;
	weight = 1.0;
	std::auto_ptr<HepMC::GenEvent> hadronLevel(hadronisation->hadronize());

	if (!hadronLevel.get()) {
		if (matchingDone) {
			if (weight == 0.0)
				partonLevel->count(LHERunInfo::kSelected);
			else
				partonLevel->count(LHERunInfo::kKilled, weight);
		} else
			partonLevel->count(LHERunInfo::kTried);
	}

	if (!matchingDone && jetMatching.get() && hadronLevel.get())
		weight = matching(hadronLevel.get(), false);

	if (weight == 0.0) {
		edm::LogInfo("Generator|LHEInterface")
			<< "Event got rejected by the "
			   "jet matching." << std::endl;

		if (hadronLevel.get()) {
			partonLevel->count(LHERunInfo::kSelected);
			hadronLevel.reset();
		}
	}

	if (!hadronLevel.get()) {
		event.put(result);
		return;
	}

	partonLevel->count(LHERunInfo::kAccepted, weight);

	hadronLevel->set_event_number(++index);

	if (eventsToPrint) {
		eventsToPrint--;
		hadronLevel->print();
	}

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

double LHEProducer::matching(const HepMC::GenEvent *event, bool shower) const
{
	if (!jetMatching.get())
		return 1.0;

	return jetMatching->match(partonLevel->asHepMCEvent().get(),
	                          event, shower);
}

bool LHEProducer::showeredEvent(const boost::shared_ptr<HepMC::GenEvent> &event)
{
	weight = matching(event.get(), true);
	matchingDone = true;
	return weight == 0.0;
}

DEFINE_ANOTHER_FWK_MODULE(LHEProducer);
