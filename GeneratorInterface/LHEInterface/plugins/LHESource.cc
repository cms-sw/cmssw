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
#include "GeneratorInterface/LHEInterface/interface/JetClustering.h"

using namespace lhef;

class LHESource : public edm::GeneratedInputSource {
    public:
	LHESource(const edm::ParameterSet &params,
	          const edm::InputSourceDescription &desc);
	virtual ~LHESource();

    private:
	virtual void endJob();
	virtual void endRun(edm::Run &run);
	virtual bool produce(edm::Event &event);

	std::auto_ptr<LHEReader>	reader;
	unsigned int			skipEvents;
	unsigned int			eventsToPrint;
	std::auto_ptr<Hadronisation>	hadronisation;
	std::auto_ptr<JetInput>		jetInput;
	std::auto_ptr<JetClustering>	jetClustering;

	const double			extCrossSect;
	const double			extFilterEff;
};

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	GeneratedInputSource(params, desc),
	reader(new LHEReader(params)),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0)),
	eventsToPrint(params.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	hadronisation(Hadronisation::create(
		params.getParameter<edm::ParameterSet>("hadronisation"))),
	extCrossSect(params.getUntrackedParameter<double>("crossSection", -1.0)),
	extFilterEff(params.getUntrackedParameter<double>("filterEfficiency", -1.0))
{
	if (params.exists("jetClustering")) {
		edm::ParameterSet jetParams = params.getUntrackedParameter<edm::ParameterSet>("jetClustering");
		jetInput.reset(new JetInput(jetParams));
		jetClustering.reset(new JetClustering(jetParams));
	}

	produces<edm::HepMCProduct>();
	produces<edm::GenInfoProduct, edm::InRun>();
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
	double crossSection = hadronisation->getCrossSection();

	std::auto_ptr<edm::GenInfoProduct> genInfoProd(new edm::GenInfoProduct);

	genInfoProd->set_cross_section(crossSection);
	genInfoProd->set_external_cross_section(extCrossSect);
	genInfoProd->set_filter_efficiency(extFilterEff);

	run.put(genInfoProd);
}

bool LHESource::produce(edm::Event &event)
{
	std::auto_ptr<HepMC::GenEvent> hadronLevel;

	while(true) {
	 	boost::shared_ptr<LHEEvent> partonLevel = reader->next();
		if (!partonLevel.get())
			return false;

		hadronisation->setEvent(partonLevel);

		hadronLevel = hadronisation->hadronize();

		if (!hadronLevel.get())
			continue;

		if (skipEvents > 0) {
			skipEvents--;
			continue;
		}

		break;
	}

	hadronLevel->set_event_number(numberEventsInRun()
	                              - remainingEvents() - 1);

	if (eventsToPrint) {
		eventsToPrint--;
		hadronLevel->print();
	}

	if (jetClustering.get()) {
		JetClustering::ParticleVector input =
					(*jetInput)(hadronLevel.get());
		std::vector<JetClustering::Jet> jets =
					(*jetClustering)(input);

		std::cout << "===== " << jets.size() << " jets:" << std::endl;
		for(std::vector<JetClustering::Jet>::const_iterator iter = jets.begin();
		    iter != jets.end(); ++iter) {
			std::cout << "* pt = " << iter->pt()
			          << ", eta = " << iter->eta()
			          << ", phi = " << iter->phi()
			          << std::endl;
			for(JetClustering::ParticleVector::const_iterator c = iter->constituents().begin();
			    c != iter->constituents().end(); c++)
				std::cout << "\tpid = " << (*c)->pdg_id()
				          << ", pt = " << (*c)->momentum().perp()
				          << ", eta = " << (*c)->momentum().eta()
				          << ", phi = " << (*c)->momentum().phi()
				          << std::endl;
		}
	}

	std::auto_ptr<edm::HepMCProduct> result(new edm::HepMCProduct);
	result->addHepMCData(hadronLevel.release());
	event.put(result);

	return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(LHESource);
