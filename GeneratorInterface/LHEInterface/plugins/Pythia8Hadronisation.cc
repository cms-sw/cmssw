#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <memory>
#include <assert.h>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>

#include <Pythia.h>
#include <LesHouches.h>
#include <HepMCInterface.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

using namespace Pythia8;

namespace lhef {

class Pythia8Hadronisation : public Hadronisation {
    public:
	Pythia8Hadronisation(const edm::ParameterSet &params);
	~Pythia8Hadronisation();

    private:
	void doInit();
	std::auto_ptr<HepMC::GenEvent> doHadronisation();
	void newRunInfo(const boost::shared_ptr<LHERunInfo> &runInfo);

	const int				pythiaPylistVerbosity;
	int					maxEventsToPrint;
	std::vector<std::string>		paramLines;

	class LHAupLesHouches;

	std::auto_ptr<Pythia>			pythia;
	std::auto_ptr<LHAupLesHouches>		lhaUP;
	std::auto_ptr<HepMC::I_Pythia8>		conv;
};

class Pythia8Hadronisation::LHAupLesHouches : public LHAup {
    public:
	LHAupLesHouches(Hadronisation *hadronisation) :
					hadronisation(hadronisation) {}

	void loadRunInfo(const boost::shared_ptr<LHERunInfo> &runInfo)
	{ this->runInfo = runInfo; }

	void loadEvent(const boost::shared_ptr<LHEEvent> &event)
	{ this->event = event; }

    private:

	bool setInit();
	bool setEvent(int idProcIn);

	Hadronisation			*hadronisation;
	boost::shared_ptr<LHERunInfo>	runInfo;
	boost::shared_ptr<LHEEvent>	event;
};

bool Pythia8Hadronisation::LHAupLesHouches::setInit()
{
	if (!runInfo)
		return false;
	const HEPRUP &heprup = *runInfo->getHEPRUP();

	setBeamA(heprup.IDBMUP.first, heprup.EBMUP.first,
	         heprup.PDFGUP.first, heprup.PDFSUP.first);
	setBeamB(heprup.IDBMUP.second, heprup.EBMUP.second,
	         heprup.PDFGUP.second, heprup.PDFSUP.second);
	setStrategy(heprup.IDWTUP);

	for(int i = 0; i < heprup.NPRUP; i++)
		addProcess(heprup.LPRUP[i], heprup.XSECUP[i],
		           heprup.XERRUP[i], heprup.XMAXUP[i]);

	hadronisation->onInit().emit();

	runInfo.reset();
	return true;
}

bool Pythia8Hadronisation::LHAupLesHouches::setEvent(int inProcId)
{
	if (!event)
		return false;
	const HEPEUP &hepeup = *event->getHEPEUP();

	setProcess(hepeup.IDPRUP, hepeup.XWGTUP, hepeup.SCALUP,
	           hepeup.AQEDUP, hepeup.AQCDUP);

	for(int i = 0; i < hepeup.NUP; i++)
		addParticle(hepeup.IDUP[i], hepeup.ISTUP[i],
		            hepeup.MOTHUP[i].first, hepeup.MOTHUP[i].second,
		            hepeup.ICOLUP[i].first, hepeup.ICOLUP[i].second,
		            hepeup.PUP[i][0], hepeup.PUP[i][1],
		            hepeup.PUP[i][2], hepeup.PUP[i][3],
		            hepeup.PUP[i][4], hepeup.VTIMUP[i],
		            hepeup.SPINUP[i]);

	const LHEEvent::PDF *pdf = event->getPDF();
	if (pdf)
		this->setPdf(pdf->id.first, pdf->id.second,
		             pdf->x.first, pdf->x.second,
		             pdf->scalePDF,
		             pdf->xPDF.first, pdf->xPDF.second, true);

	hadronisation->onBeforeHadronisation().emit();

	event.reset();
	return true;
}

Pythia8Hadronisation::Pythia8Hadronisation(const edm::ParameterSet &params) :
	Hadronisation(params),
	pythiaPylistVerbosity(params.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
	maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0))
{
	std::vector<std::string> setNames =
		params.getParameter<std::vector<std::string> >("parameterSets");

	for(std::vector<std::string>::const_iterator iter = setNames.begin();
	    iter != setNames.end(); ++iter) {
		std::vector<std::string> lines =
			params.getParameter< std::vector<std::string> >(*iter);

		for(std::vector<std::string>::const_iterator line = lines.begin();
		    line != lines.end(); ++line )
			if (line->substr(0, 14) == "Random:setSeed" ||
			    line->substr(0, 11) == "Random:seed")
				throw cms::Exception("PythiaError")
					<< "Attempted to set random number"
					   " using Pythia command 'MRPY(1)'."
					   " Please use the"
					   " RandomNumberGeneratorService."
					<< std::endl;

		std::copy(lines.begin(), lines.end(),
		          std::back_inserter(paramLines));
	}
}

Pythia8Hadronisation::~Pythia8Hadronisation()
{
}

void Pythia8Hadronisation::doInit()
{
	pythia.reset(new Pythia);
	lhaUP.reset(new LHAupLesHouches(this));
	conv.reset(new HepMC::I_Pythia8);

	for(std::vector<std::string>::const_iterator iter = paramLines.begin();
	    iter != paramLines.end(); ++iter)
		if (!pythia->readString(*iter))
			throw cms::Exception("PythiaError")
				<< "Pythia did not accept \""
				<< *iter << "\"." << std::endl;

	edm::Service<edm::RandomNumberGenerator> rng;
	std::ostringstream ss;
	ss << "Random:seed = " << rng->mySeed();
	pythia->readString(ss.str());
	pythia->readString("Random:setSeed = on");
}

// naive Pythia8 HepMC status fixup
static int getStatus(const HepMC::GenParticle *p)
{
	int status = p->status();
	if (status > 0)
		return status;
	else if (status > -30 && status < 0)
		return 3;
	else
		return 2;
}

std::auto_ptr<HepMC::GenEvent> Pythia8Hadronisation::doHadronisation()
{
	lhaUP->loadEvent(getRawEvent());
	if (!pythia->next())
		throw cms::Exception("PythiaError")
			<< "Pythia did not want to process event."
			<< std::endl;

	std::auto_ptr<HepMC::GenEvent> event(new HepMC::GenEvent);
	conv->fill_next_event(pythia->event, event.get());

	for(HepMC::GenEvent::particle_iterator iter = event->particles_begin();
	    iter != event->particles_end(); iter++)
		(*iter)->set_status(getStatus(*iter));

	event->set_signal_process_id(pythia->info.code());
	event->set_event_scale(pythia->info.pTHat());

	if (maxEventsToPrint > 0) {
		maxEventsToPrint--;
		if (pythiaPylistVerbosity)
			pythia->event.list(std::cout);
	}

	return event;
}

void Pythia8Hadronisation::newRunInfo(
				const boost::shared_ptr<LHERunInfo> &runInfo)
{
	lhaUP->loadRunInfo(runInfo);
	pythia->init(lhaUP.get());
}

DEFINE_LHE_HADRONISATION_PLUGIN(Pythia8Hadronisation);

} // namespace lhef
