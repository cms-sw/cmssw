#ifndef CondFormats_BeamSpotObjects_SimBeamSpotHLLHCObjects_h
#define CondFormats_BeamSpotObjects_SimBeamSpotHLLHCObjects_h

/** \class SimBeamSpotHLLHCObjects
 *
 * Provide the vertex smearing parameters from DB
 *
 * This Object contains the parameters needed by the HLLHCEvtVtxGenerator generator:
 *   Parameters used:
 *      - fMeanX, fMeanY, fMeanZ
 *      - fEProton, fCrabFrequency, fRF800
 *      - fCrossingAngle
 *      - fCrabbingAngleCrossing, fCrabbingAngleSeparation
 *      - fBetaCrossingPlane, fBetaSeparationPlane
 *      - fHorizontalEmittance, fVerticalEmittance
 *      - fBunchLength
 *      - fTimeOffset
 *
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include <sstream>

class SimBeamSpotHLLHCObjects {
public:
  /// default constructor
  SimBeamSpotHLLHCObjects() {
    fMeanX = 0.0;
    fMeanY = 0.0;
    fMeanZ = 0.0;
    fEProton = 0.0;
    fCrabFrequency = 0.0;
    fRF800 = 0.0;
    fCrossingAngle = 0.0;
    fCrabbingAngleCrossing = 0.0;
    fCrabbingAngleSeparation = 0.0;
    fBetaCrossingPlane = 0.0;
    fBetaSeparationPlane = 0.0;
    fHorizontalEmittance = 0.0;
    fVerticalEmittance = 0.0;
    fBunchLength = 0.0;
    fTimeOffset = 0.0;
  };

  virtual ~SimBeamSpotHLLHCObjects() {}

  /// set meanX, meanY, meanZ
  void setMeanX(double val) { fMeanX = val; }
  void setMeanY(double val) { fMeanY = val; }
  void setMeanZ(double val) { fMeanZ = val; }
  /// set EProton, fCrabFrequency, RF800
  void setEProton(double val) { fEProton = val; }
  void setCrabFrequency(double val) { fCrabFrequency = val; }
  void setRF800(double val) { fRF800 = val; }
  /// set Crossing and Crabbing angles
  void setCrossingAngle(double val) { fCrossingAngle = val; }
  void setCrabbingAngleCrossing(double val) { fCrabbingAngleCrossing = val; }
  void setCrabbingAngleSeparation(double val) { fCrabbingAngleSeparation = val; }
  /// set BetaStar and Emittance
  void setBetaCrossingPlane(double val) { fBetaCrossingPlane = val; }
  void setBetaSeparationPlane(double val) { fBetaSeparationPlane = val; }
  void setHorizontalEmittance(double val) { fHorizontalEmittance = val; }
  void setVerticalEmittance(double val) { fVerticalEmittance = val; }
  /// set BunchLength and TimeOffset
  void setBunchLength(double val) { fBunchLength = val; }
  void setTimeOffset(double val) { fTimeOffset = val; }

  /// get meanX, meanY, meanZ position
  double meanX() const { return fMeanX; }
  double meanY() const { return fMeanY; }
  double meanZ() const { return fMeanZ; }
  /// get EProton, fCrabFrequency, RF800
  double eProton() const { return fEProton; }
  double crabFrequency() const { return fCrabFrequency; }
  double rf800() const { return fRF800; }
  /// set Crossing and Crabbing angles
  double crossingAngle() const { return fCrossingAngle; }
  double crabbingAngleCrossing() const { return fCrabbingAngleCrossing; }
  double crabbingAngleSeparation() const { return fCrabbingAngleSeparation; }
  /// get BetaStar and Emittance
  double betaCrossingPlane() const { return fBetaCrossingPlane; }
  double betaSeparationPlane() const { return fBetaSeparationPlane; }
  double horizontalEmittance() const { return fHorizontalEmittance; }
  double verticalEmittance() const { return fVerticalEmittance; }
  /// get BunchLength and TimeOffset
  double bunchLenght() const { return fBunchLength; }
  double timeOffset() const { return fTimeOffset; }

  /// print sim beam spot parameters
  void print(std::stringstream& ss) const;

private:
  double fMeanX, fMeanY, fMeanZ;
  double fEProton, fCrabFrequency, fRF800;
  double fCrossingAngle, fCrabbingAngleCrossing, fCrabbingAngleSeparation;
  double fBetaCrossingPlane, fBetaSeparationPlane;
  double fHorizontalEmittance, fVerticalEmittance;
  double fBunchLength, fTimeOffset;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, SimBeamSpotHLLHCObjects beam);

#endif
