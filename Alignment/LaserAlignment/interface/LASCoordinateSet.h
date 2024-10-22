

#ifndef __LASCOORDINATESET_H
#define __LASCOORDINATESET_H

#include <iostream>

///
/// container for phi, x, y coordinates
/// and their errors
///
class LASCoordinateSet {
public:
  LASCoordinateSet() : phi(0.), phiError(0.), r(0.), rError(0.), z(0.), zError(0.) {}
  LASCoordinateSet(double, double, double, double, double, double);

  void GetCoordinates(double&, double&, double&, double&, double&, double&) const;
  double GetPhi(void) const { return phi; }
  double GetPhiError(void) const { return phiError; }
  double GetR(void) const { return r; }
  double GetRError(void) const { return rError; }
  double GetZ(void) const { return z; }
  double GetZError(void) const { return zError; }

  void SetCoordinates(double, double, double, double, double, double);
  void SetErrors(double, double, double);
  void SetPhi(double aPhi) { phi = aPhi; }
  void SetPhi(double aPhi, double aPhiError) {
    phi = aPhi;
    phiError = aPhiError;
  }
  void SetPhiError(double aPhiError) { phiError = aPhiError; }
  void SetR(double aR) { r = aR; }
  void SetR(double aR, double aRError) {
    r = aR;
    rError = aRError;
  }
  void SetRError(double aRError) { rError = aRError; }
  void SetZ(double aZ) { z = aZ; }
  void SetZ(double aZ, double aZError) {
    z = aZ;
    zError = aZError;
  }
  void SetZError(double aZError) { zError = aZError; }

  void Dump(void);

private:
  double phi;
  double phiError;
  double r;
  double rError;
  double z;
  double zError;
};

#endif
