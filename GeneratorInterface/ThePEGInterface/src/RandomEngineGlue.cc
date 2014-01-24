#include <string>

#include <CLHEP/Random/RandomEngine.h>

#include <ThePEG/Interface/ClassDocumentation.h>
#include <ThePEG/Interface/InterfacedBase.h>
#include <ThePEG/Interface/Parameter.h>
#include <ThePEG/Utilities/ClassTraits.h>

#include <ThePEG/Repository/StandardRandom.h>

#include "GeneratorInterface/ThePEGInterface/interface/RandomEngineGlue.h"

#include "FWCore/Utilities/interface/EDMException.h"

using namespace ThePEG;

RandomEngineGlue::RandomEngineGlue() :
	randomEngine(nullptr)
{
}

RandomEngineGlue::~RandomEngineGlue()
{
}

void RandomEngineGlue::flush()
{
	RandomGenerator::flush();
	gaussSaved = false;
}

void RandomEngineGlue::fill()
{
        if(randomEngine == nullptr) {
          throw edm::Exception(edm::errors::LogicError)
            << "The PEG code attempted to a generate random number while\n"
            << "the engine pointer was null. This might mean that the code\n"
            << "was tried to generate a random number outside the event and\n"
            << "beginLuminosityBlock methods, which is not allowed.\n";
        }
	nextNumber = theNumbers.begin();
	for(RndVector::iterator it = nextNumber; it != theNumbers.end(); ++it)
		*it = randomEngine->flat();
}

void RandomEngineGlue::setSeed(long seed)
{
	// we ignore this, CMSSW overrides the seed from ThePEG
}

void RandomEngineGlue::doinit() throw(InitException)
{
	RandomGenerator::doinit();

	boost::shared_ptr<Proxy> proxy = Proxy::find(proxyID);
	if (!proxy)
		throw InitException();

	proxy->instance = this;
        randomEngine = proxy->getRandomEngine();
	flush();
}

ClassDescription<RandomEngineGlue> RandomEngineGlue::initRandomEngineGlue;

void RandomEngineGlue::Init() {
	typedef Proxy::ProxyID ProxyID;

	static ClassDocumentation<RandomEngineGlue> documentation
		("Interface to the CMSSW RandomNumberEngine.");
	static Parameter<RandomEngineGlue, ProxyID> interfaceProxyID
		("ProxyID", "The ProxyID.",
		 &RandomEngineGlue::proxyID, ProxyID(),
		 ProxyID(), ProxyID(), false, false, false);

	interfaceProxyID.rank(11);
}
