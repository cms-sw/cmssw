#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <set>

#include <boost/shared_ptr.hpp>

#include <ThePEG/Interface/ClassDocumentation.h>
#include <ThePEG/Interface/InterfacedBase.h>
#include <ThePEG/Interface/Parameter.h>
#include <ThePEG/Utilities/ClassTraits.h>

#include <ThePEG/LesHouches/LesHouches.h>
#include <ThePEG/LesHouches/LesHouchesReader.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEProxy.h"

using namespace lhef;

namespace ThePEG {

class LesHouchesInterface : public LesHouchesReader {
    public:
	LesHouchesInterface();
	~LesHouchesInterface() override;

	static void Init();

    protected:		
	IBPtr clone() const override { return new_ptr(*this); }
	IBPtr fullclone() const override { return new_ptr(*this); }

    private:
	void open() override;
	void close() override;
	long scan() override;

	virtual double eventWeight();
	virtual double reweight();
	double getEvent() override;
	bool doReadEvent() override;

	LHEProxy::ProxyID	proxyID;
	bool			initialized;

	static ClassDescription<LesHouchesInterface> initLesHouchesInterface;
};

// registration with ThePEG plugin system

template<>
struct BaseClassTrait<LesHouchesInterface, 1> : public ClassTraitsType {
	/** Typedef of the base class of LesHouchesInterface. */
	typedef LesHouchesReader NthBase;
};

template<>
struct ClassTraits<LesHouchesInterface> :
			public ClassTraitsBase<LesHouchesInterface> {
	static string className() { return "ThePEG::LesHouchesInterface"; }
	static string library() { return "pluginGeneratorInterfaceThePEGHadronisation.so"; }
};

} // namespace ThePEG

using namespace ThePEG;

LesHouchesInterface::LesHouchesInterface() :
	initialized(false)
{
}

LesHouchesInterface::~LesHouchesInterface()
{
}

long LesHouchesInterface::scan()
{
	return 1000000000;
}

void LesHouchesInterface::open()
{
	const LHERunInfo *runInfo = LHEProxy::find(proxyID)->getRunInfo().get();
	const lhef::HEPRUP &orig = *runInfo->getHEPRUP();

	heprup.IDBMUP	= orig.IDBMUP;
	heprup.EBMUP	= orig.EBMUP;
	heprup.PDFGUP	= orig.PDFGUP;
	heprup.PDFSUP	= orig.PDFSUP;
	heprup.NPRUP	= orig.NPRUP;
	heprup.LPRUP	= orig.LPRUP;
	heprup.XSECUP	= orig.XSECUP;
	heprup.XERRUP	= orig.XERRUP;
	heprup.XMAXUP	= orig.XMAXUP;

	// We are cheating here, ThePEG does not need to know the real
	// weighting method, as it complains about anything but 3 anyway.
	// We just need to trick ThePEG into just processing each event
	// passed without doing anything fancy with it (and shut up).
	heprup.IDWTUP = 1;
}

void LesHouchesInterface::close()
{
}

double LesHouchesInterface::eventWeight()
{
	return 1.0;
}

// no reweighting
double LesHouchesInterface::reweight()
{
	preweight = 1.0;
	return 1.0;
}

// overwrite parent routine doing fancy stuff and just pass the event
double LesHouchesInterface::getEvent()
{
	reset();

	if (!doReadEvent())
		return 0.0;

	if (!initialized && !checkPartonBin())
		throw cms::Exception("ThePEGLesHouchesInterface")
			<< "Found event which cannot be handled by "
			<< "the assigned PartonExtractor." << std::endl;
	initialized = true;

	fillEvent();
	getSubProcess();

	return 1.0;
}

bool LesHouchesInterface::doReadEvent()
{
	reset();

	boost::shared_ptr<LHEEvent> event =
				LHEProxy::find(proxyID)->releaseEvent();
	if (!event)
		throw Stop();

	hepeup.XPDWUP.first = hepeup.XPDWUP.second = 0;

	const lhef::HEPEUP &orig = *event->getHEPEUP();

	hepeup.NUP	= orig.NUP;
	hepeup.IDPRUP	= orig.IDPRUP;
	hepeup.SCALUP	= orig.SCALUP;
	hepeup.AQEDUP	= orig.AQEDUP;
	hepeup.AQCDUP	= orig.AQCDUP;
        
        //workaround, since Herwig++ is not passing LHE weights to the hepmc product anyways
        //as currently run in CMSSW
        hepeup.XWGTUP   = 1.0;
        
	hepeup.resize();

	std::copy(orig.IDUP.begin(), orig.IDUP.end(), hepeup.IDUP.begin());
	hepeup.ISTUP	= orig.ISTUP;
	hepeup.MOTHUP	= orig.MOTHUP;
	hepeup.ICOLUP	= orig.ICOLUP;
	hepeup.VTIMUP	= orig.VTIMUP;
	hepeup.SPINUP	= orig.SPINUP;

	for(int i = 0; i < hepeup.NUP; i++)
		std::copy(&orig.PUP[i].x[0], &orig.PUP[i].x[5],
		          hepeup.PUP[i].begin());

	fillEvent();

	return true;
}

// register with ThePEG plugin system

ClassDescription<LesHouchesInterface> LesHouchesInterface::initLesHouchesInterface;

void LesHouchesInterface::Init() {
	typedef LHEProxy::ProxyID ProxyID;

	static ClassDocumentation<LesHouchesInterface> documentation
		("ThePEG::LesHouchesInterface interfaces with LHEInterface");

	static Parameter<LesHouchesInterface, ProxyID> interfaceProxyID
		("ProxyID", "The ProxyID.",
		 &LesHouchesInterface::proxyID, ProxyID(),
		 ProxyID(), ProxyID(), false, false, false);

	interfaceProxyID.rank(11);
}
