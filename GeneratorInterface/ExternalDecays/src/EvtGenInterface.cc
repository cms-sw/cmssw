
#include "GeneratorInterface/ExternalDecays/interface/EvtGenInterface.h"

#include "HepMC/GenEvent.h"

using namespace gen;
using namespace edm;

EvtGenInterface::EvtGenInterface( const ParameterSet& pset )
{
} 

EvtGenInterface::~EvtGenInterface()
{
}

bool EvtGenInterface::decay( HepMC::GenEvent* evt )
{
   return true;
}
