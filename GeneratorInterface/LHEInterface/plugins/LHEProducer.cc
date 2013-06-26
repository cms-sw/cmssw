#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>
#include <sigc++/signal.h>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatchingMLM.h"
#include "GeneratorInterface/LHEInterface/interface/BranchingRatios.h"

using namespace lhef;

class LHEProducer : public edm::one::EDFilter<edm::EndRunProducer,
                                              edm::one::WatchRuns> {
    public:
	explicit LHEProducer(const edm::ParameterSet &params);
	virtual ~LHEProducer();

    protected:
        virtual void beginJob() override;
	virtual void endJob() override;
	virtual void beginRun(edm::Run const& run, const edm::EventSetup &es) override;
	virtual void endRun(edm::Run const&run, const edm::EventSetup &es) override;
	virtual void endRunProduce(edm::Run &run, const edm::EventSetup &es) override;
	virtual bool filter(edm::Event &event, const edm::EventSetup &es) override;

    private:
	double matching(const HepMC::GenEvent *event, bool shower) const;

	bool showeredEvent(const boost::shared_ptr<HepMC::GenEvent> &event);
	void onInit();
	void onBeforeHadronisation();

	unsigned int			eventsToPrint;
	std::vector<int>		removeResonances;
	std::auto_ptr<Hadronisation>	hadronisation;
	std::auto_ptr<JetMatching>	jetMatching;

	double				extCrossSect;
	double				extFilterEff;
	bool				matchSummary;

	boost::shared_ptr<LHEEvent>	partonLevel;
	boost::shared_ptr<LHERunInfo>	runInfo;
	unsigned int			index;
	bool				matchingDone;
	double				weight;
	BranchingRatios			branchingRatios;
};

LHEProducer::LHEProducer(const edm::ParameterSet &params) :
	eventsToPrint(params.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	extCrossSect(params.getUntrackedParameter<double>("crossSection", -1.0)),
	extFilterEff(params.getUntrackedParameter<double>("filterEfficiency", -1.0)),
	matchSummary(false)
{
	hadronisation = Hadronisation::create(
		params.getParameter<edm::ParameterSet>("hadronisation"));

	if (params.exists("removeResonances"))
		removeResonances =
			params.getParameter<std::vector<int> >(
							"removeResonances");

	std::set<std::string> matchingCapabilities;
	if (params.exists("jetMatching")) {
		edm::ParameterSet jetParams =
			params.getUntrackedParameter<edm::ParameterSet>(
								"jetMatching");
		jetMatching = JetMatching::create(jetParams);

		matchingCapabilities = jetMatching->capabilities();
		hadronisation->matchingCapabilities(matchingCapabilities);
	}

	produces<edm::HepMCProduct>();
	produces<GenEventInfoProduct>();
	produces<GenRunInfoProduct, edm::InRun>();

	if (jetMatching.get()) {
		if (params.getUntrackedParameter<bool>(
					"preferShowerVetoCallback", true))
			hadronisation->onShoweredEvent().connect(
				sigc::mem_fun(*this,
				              &LHEProducer::showeredEvent));
		hadronisation->onInit().connect(
				sigc::mem_fun(*this, &LHEProducer::onInit));
		hadronisation->onBeforeHadronisation().connect(
			sigc::mem_fun(*this,
			              &LHEProducer::onBeforeHadronisation));

		matchSummary = matchingCapabilities.count("matchSummary");
		if (matchSummary) {
			produces< std::vector<double> >("matchDeltaR");
			produces< std::vector<double> >("matchDeltaPRel");
		}
	}

	// force total branching ratio for QCD/QED to 1
	for(int i = 0; i < 6; i++)
		branchingRatios.set(i);
	for(int i = 9; i < 23; i++)
		branchingRatios.set(i);
}

LHEProducer::~LHEProducer()
{
}

void LHEProducer::beginJob()
{
	hadronisation->init();
}

void LHEProducer::endJob()
{
	hadronisation.reset();
	jetMatching.reset();
}

void LHEProducer::beginRun(edm::Run const& run, const edm::EventSetup &es)
{
	edm::Handle<LHERunInfoProduct> product;
	run.getByLabel("source", product);

	runInfo.reset(new LHERunInfo(*product));
	index = 0;
}
void LHEProducer::endRun(edm::Run const& run, const edm::EventSetup &es)
{
}

void LHEProducer::endRunProduce(edm::Run &run, const edm::EventSetup &es)
{
	hadronisation->statistics();

	LHERunInfo::XSec crossSection;
	if (runInfo) {
		crossSection = runInfo->xsec();
		runInfo->statistics();
	}

	std::auto_ptr<GenRunInfoProduct> runInfo(new GenRunInfoProduct);

	runInfo->setInternalXSec(
			GenRunInfoProduct::XSec(crossSection.value,
			                        crossSection.error));
	runInfo->setExternalXSecLO(extCrossSect);
	runInfo->setFilterEfficiency(extFilterEff);

	run.put(runInfo);

	runInfo.reset();
}

bool LHEProducer::filter(edm::Event &event, const edm::EventSetup &es)
{
	std::auto_ptr<edm::HepMCProduct> result(new edm::HepMCProduct);

	edm::Handle<LHEEventProduct> product;
	event.getByLabel("source", product);

	partonLevel.reset(new LHEEvent(runInfo, *product));
	if (!removeResonances.empty())
		partonLevel->removeResonances(removeResonances);

	if (product->pdf())
		partonLevel->setPDF(
			std::auto_ptr<LHEEvent::PDF>(
				new LHEEvent::PDF(*product->pdf())));

	hadronisation->setEvent(partonLevel);

	double br = branchingRatios.getFactor(hadronisation.get(),
	                                      partonLevel);

	matchingDone = false;
	weight = 1.0;
	std::auto_ptr<HepMC::GenEvent> hadronLevel(hadronisation->hadronize());

	if (!hadronLevel.get()) {
		if (matchingDone) {
			if (weight == 0.0)
				partonLevel->count(LHERunInfo::kSelected, br);
			else
				partonLevel->count(LHERunInfo::kKilled,
				                   br, weight);
		} else
			partonLevel->count(LHERunInfo::kTried, br);
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
		std::auto_ptr<GenEventInfoProduct> info(
						new GenEventInfoProduct);
		event.put(info);
		return false;
	}

	partonLevel->count(LHERunInfo::kAccepted, br, weight);

	hadronLevel->set_event_number(++index);

	if (eventsToPrint) {
		eventsToPrint--;
		hadronLevel->print();
	}

	std::auto_ptr<GenEventInfoProduct> info(
				new GenEventInfoProduct(hadronLevel.get()));
	result->addHepMCData(hadronLevel.release());
	event.put(result);
	event.put(info);

	if (jetMatching.get() && matchSummary) {
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
			matchDeltaPRel->push_back(iter->jet.rho() /
			                          iter->parton.rho() - 1.0);
		}

		event.put(matchDeltaR, "matchDeltaR");
		event.put(matchDeltaPRel, "matchDeltaPRel");
	}

	return true;
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

void LHEProducer::onInit()
{
	jetMatching->init(runInfo);
}

void LHEProducer::onBeforeHadronisation()
{
	jetMatching->beforeHadronisation(partonLevel);
}

DEFINE_FWK_MODULE(LHEProducer);

DEFINE_LHE_JETMATCHING_PLUGIN(JetMatchingMLM);
