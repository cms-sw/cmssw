// stack macro
// Pasquale Noli
#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zmm2hlt() {
  makePlots("goodZToMuMu2HLTPlots/zMass", "events/GeV/c^{2}", 1, "zmm2hlt.eps", 
	    0.0001);
}
