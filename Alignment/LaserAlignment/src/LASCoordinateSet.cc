

#include "Alignment/LaserAlignment/interface/LASCoordinateSet.h"

///
/// convenience functions for the coordinates;
/// the rest is declared inline
///


///
///
///
LASCoordinateSet::LASCoordinateSet( double aPhi, double aPhiError, double aR, double aRError, double aZ, double aZError ) {
  
  phi = aPhi;
  phiError = aPhiError;
  r = aR;
  rError = aRError;
  z = aZ;
  zError = aZError;

}





///
///
///
void LASCoordinateSet::GetCoordinates( double& aPhi, double& aPhiError, double& aR, double& aRError, double& aZ, double& aZError ) const {

  aPhi = phi;
  aPhiError = phiError;
  aR = r;
  aRError = rError;
  aZ = z;
  aZError = zError;

}





///
///
///
void LASCoordinateSet::SetCoordinates( double aPhi, double aPhiError, double aR, double aRError, double aZ, double aZError ) {
  
  phi = aPhi;
  phiError = aPhiError;
  r = aR;
  rError = aRError;
  z = aZ;
  zError = aZError; 

}





///
///
///
void LASCoordinateSet::SetErrors( double aPhiError, double aRError, double aZError ) {

  phiError = aPhiError;
  rError = aRError;
  zError = aZError;

}





///
///
///
void LASCoordinateSet::Dump( void ) {

  std::cout << " [LASCoordinateSet::Dump] -- phi: " << phi << ", phiE: " << phiError
	    << ", r: " << r << ", rE: " << rError << ", z: " << z << ", zE: " << zError << " ." << std::endl;

}
