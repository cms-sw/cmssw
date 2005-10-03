#ifndef PHYSICSTOOLS_HELICITYANGLE_H
#define PHYSICSTOOLS_HELICITYANGLE_H

namespace aod {
  class Candidate;
}

double helicityAngle( const aod::Candidate & c );

#endif
