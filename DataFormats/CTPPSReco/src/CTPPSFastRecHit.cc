#include "DataFormats/CTPPSReco/interface/CTPPSFastRecHit.h"
#include <ostream>

std::ostream & operator<<(std::ostream & o, const CTPPSFastRecHit & hit) 
{ return o << hit.detUnitId() << " " << hit.entryPoint() << " " << hit.tof() << "  "  << hit.cellId(); }

