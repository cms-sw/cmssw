#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

#include <iostream>

void SimBeamSpotObjects::print(std::stringstream& ss) const {
	  ss << "-----------------------------------------------------\n"
	     <<fX0<<std::endl;
}

std::ostream& operator<< ( std::ostream& os, SimBeamSpotObjects beam ) {
  std::stringstream ss;
  beam.print(ss);
  os << ss.str();
  return os;
}
