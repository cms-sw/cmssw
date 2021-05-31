#ifndef SIMBEAMSPOTOBJECTS_H
#define SIMBEAMSPOTOBJECTS_H

/** \class SimBeamSpotObjects
 *
 * provide the vertex smearing parameters from DB
 *
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include <sstream>

class SimBeamSpotObjects {
public:
  SimBeamSpotObjects(){};
  virtual ~SimBeamSpotObjects(){};

  double fX0, fY0, fZ0;
  double fSigmaZ;
  double fbetastar, femittance;
  double fPhi, fAlpha;
  double fTimeOffset;

  void print(std::stringstream& ss) const;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, SimBeamSpotObjects beam);

#endif
