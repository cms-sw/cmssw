#ifndef gen_RNDMEngineAccess_h
#define gen_RNDMEngineAccess_h

#include <CLHEP/Random/RandomEngine.h>

namespace gen {

	CLHEP::HepRandomEngine& getEngineReference();

} // namespace gen

#endif
