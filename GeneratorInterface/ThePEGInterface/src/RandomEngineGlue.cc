#include <string>

#include <CLHEP/Random/RandomEngine.h>

#include <ThePEG/Interface/ClassDocumentation.h>
#include <ThePEG/Interface/InterfacedBase.h>
#include <ThePEG/Interface/Parameter.h>
#include <ThePEG/Utilities/ClassTraits.h>

#include <ThePEG/Repository/StandardRandom.h>

#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "GeneratorInterface/ThePEGInterface/interface/RandomEngineGlue.h"

using namespace ThePEG;

RandomEngineGlue::RandomEngineGlue() :
	randomEngine(&gen::getEngineReference())
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
