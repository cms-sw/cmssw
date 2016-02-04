#ifndef MagneticFieldVolume_H
#define MagneticFieldVolume_H

class MagneticFieldVolume {
public:

  /** Position and rotation of field parametrisation reference frame
   *  with respect to the global frame.
   */
  PositionType position() const;
  RotationType rotation() const;

  // toGlobal and toLocal available from base class...

  /** Returns the field vector in the local frame, at local position p
   */
  LocalVector  valueInTesla( const LocalPoint& p) const;

  /** Returns the field vector in the global frame, at global position p
   */
  GlobalVector valueInTesla( const GlobalPoint& p) const;

  /** Returns the maximal order of available derivatives.
   *  Returns 0 if derivatives are not available.
   */
  int hasDerivatives() const;

  /** Returns the Nth spacial derivative of the field in the local frame.
   */
  LocalVector  derivativeInTeslaPerMeter( const LocalPoint& p, int N) const;


};

#endif
