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
  /// default constructor
  SimBeamSpotObjects() {
    fX0 = 0.0;
    fY0 = 0.0;
    fZ0 = 0.0;
    fSigmaZ = 0.0;
    fbetastar = 0.0;
    femittance = 0.0;
    fPhi = 0.0;
    fAlpha = 0.0;
    fTimeOffset = 0.0;
  };

  virtual ~SimBeamSpotObjects(){};

  /// set X, Y, Z positions
  void setX(double val) { fX0 = val; }
  void setY(double val) { fY0 = val; }
  void setZ(double val) { fZ0 = val; }
  /// set sigmaZ
  void setSigmaZ(double val) { fSigmaZ = val; }
  /// set BetaStar and Emittance
  void setBetaStar(double val) { fbetastar = val; }
  void setEmittance(double val) { femittance = val; }
  /// set Phi, Alpha and TimeOffset
  void setPhi(double val) { fPhi = val; }
  void setAlpha(double val) { fAlpha = val; }
  void setTimeOffset(double val) { fTimeOffset = val; }

  /// get X position
  double x() const { return fX0; }
  /// get Y position
  double y() const { return fY0; }
  /// get Z position
  double z() const { return fZ0; }
  /// get sigmaZ
  double sigmaZ() const { return fSigmaZ; }
  /// get BetaStar
  double betaStar() const { return fbetastar; }
  /// get Emittance
  double emittance() const { return femittance; }
  /// get Phi
  double phi() const { return fPhi; }
  /// get Alpha
  double alpha() const { return fAlpha; }
  /// get TimeOffset
  double timeOffset() const { return fTimeOffset; }

  /// print sim beam spot parameters
  void print(std::stringstream& ss) const;

private:
  double fX0, fY0, fZ0;
  double fSigmaZ;
  double fbetastar, femittance;
  double fPhi, fAlpha;
  double fTimeOffset;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, SimBeamSpotObjects beam);

#endif
