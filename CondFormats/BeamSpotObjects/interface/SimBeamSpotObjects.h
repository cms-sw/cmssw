#ifndef SIMBEAMSPOTOBJECTS_H
#define SIMBEAMSPOTOBJECTS_H

/** \class SimBeamSpotObjects
 *
 * Provide the vertex smearing parameters from DB
 *
 * This Object contains the parameters needed by the vtx smearing functions used up to Run 3:
 *   - BetafuncEvtVtxGenerator (realistic Run1/Run2/Run3 conditions)
 *     Parameters used:
 *        - fX0, fY0, fZ0
 *        - fSigmaZ
 *        - fbetastar, femittance
 *        - fPhi, fAlpha
 *        - fTimeOffset
 *   - GaussEvtVtxGenerator (design Run1/Run2/Run3 conditions)
 *     Parameters used:
 *        - fMeanX, fMeanY, fMeanZ
 *        - fSigmaX, fSigmaY, fSigmaZ
 *        - fTimeOffset
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
    fMeanX = 0.0;
    fMeanY = 0.0;
    fMeanZ = 0.0;
    fSigmaX = -1.0;
    fSigmaY = -1.0;
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
  /// set meanX, meanY, meanZ
  void setMeanX(double val) { fMeanX = val; }
  void setMeanY(double val) { fMeanY = val; }
  void setMeanZ(double val) { fMeanZ = val; }
  /// set sigmaX, sigmaY, sigmaZ
  void setSigmaX(double val) { fSigmaX = val; }
  void setSigmaY(double val) { fSigmaY = val; }
  void setSigmaZ(double val) { fSigmaZ = val; }
  /// set BetaStar and Emittance
  void setBetaStar(double val) { fbetastar = val; }
  void setEmittance(double val) { femittance = val; }
  /// set Phi, Alpha and TimeOffset
  void setPhi(double val) { fPhi = val; }
  void setAlpha(double val) { fAlpha = val; }
  void setTimeOffset(double val) { fTimeOffset = val; }

  /// get X, Y, Z position
  double x() const { return fX0; }
  double y() const { return fY0; }
  double z() const { return fZ0; }
  /// get meanX, meanY, meanZ position
  double meanX() const { return fMeanX; }
  double meanY() const { return fMeanY; }
  double meanZ() const { return fMeanZ; }
  /// get sigmaX, sigmaY, sigmaZ
  double sigmaX() const;
  double sigmaY() const;
  double sigmaZ() const { return fSigmaZ; }
  /// get BetaStar and Emittance
  double betaStar() const { return fbetastar; }
  double emittance() const { return femittance; }
  /// get Phi, Alpha and TimeOffset
  double phi() const { return fPhi; }
  double alpha() const { return fAlpha; }
  double timeOffset() const { return fTimeOffset; }

  /// print sim beam spot parameters
  void print(std::stringstream& ss) const;

private:
  double fX0, fY0, fZ0;
  double fMeanX, fMeanY, fMeanZ;
  double fSigmaX, fSigmaY, fSigmaZ;
  double fbetastar, femittance;
  double fPhi, fAlpha;
  double fTimeOffset;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, SimBeamSpotObjects beam);

#endif
