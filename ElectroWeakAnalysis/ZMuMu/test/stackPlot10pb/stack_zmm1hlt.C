// stack macro
// Pasquale Noli
#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zmm1hlt() {
  makePlots("goodZToMuMu1HLTPlots/zMass", "events/GeV/c^{2}", 1, "zmm1hlt.eps", 
	    0.0001);
}
