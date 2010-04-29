// stack macro
// Pasquale Noli
#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zmTrk() {
  makePlots("goodZToMuMuOneTrackPlots/zMass", "events/5 GeV/c^{2}", 5, "zmTrk.eps", 
	    0.0001, true);
}
