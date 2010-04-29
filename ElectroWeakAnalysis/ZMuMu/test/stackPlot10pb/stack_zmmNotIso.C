// stack macro
// Pasquale Noli
#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zmmNotIso() {
  makePlots("nonIsolatedZToMuMuPlots/zMass", "events/5 GeV/c^{2}", 5, "zmmNotIso.eps", 
	    0.0001);
}
