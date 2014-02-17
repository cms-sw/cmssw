/**_________________________________________________________________
   class:   BeamSpot.cc
   package: DataFormats/BeamSpot
   
 A reconstructed beam spot providing position, width, slopes,
 and errors.

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpot.cc,v 1.13 2012/10/19 07:48:45 innocent Exp $

 ________________________________________________________________**/

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"

#include <iostream>


namespace reco {

  using namespace math;

  BeamSpot::BeamSpot() {
	  // initialize
	  position_ = Point(0.,0.,0.);
	  sigmaZ_ = 0.;
	  dxdz_ = 0.;
	  dydz_ = 0.;
	  BeamWidthX_ = 0.; BeamWidthY_ = 0;
	  for (int j=0; j<7; j++) {
			for (int k=j; k<7; k++) {
				error_(j,k) = 0.;
			}
	  }
	  type_ = Unknown;
	  emittanceX_ = 0;
	  emittanceY_ = 0;
	  betaStar_ = 0;
  }
   	
  const BeamSpot::Point BeamSpot::position(const double z) const {

    Point pos(x(z),y(z),z);
    return pos;

  }


  void BeamSpot::print(std::stringstream& ss) const {

    ss << "-----------------------------------------------------\n"
       << "              Beam Spot Data\n\n"
	   << " Beam type    = " << type() << "\n"
       << "       X0     = " << x0() << " +/- " << x0Error() << " [cm]\n"
       << "       Y0     = " << y0() << " +/- " << y0Error() << " [cm]\n"
       << "       Z0     = " << z0() << " +/- " << z0Error() << " [cm]\n"
       << " Sigma Z0     = " << sigmaZ() << " +/- " << sigmaZ0Error() << " [cm]\n"
       << " dxdz         = " << dxdz() << " +/- " << dxdzError() << " [radians]\n"
       << " dydz         = " << dydz() << " +/- " << dydzError() << " [radians]\n"
       << " Beam width X = " << BeamWidthX() << " +/- " << BeamWidthXError() << " [cm]\n"
	   << " Beam width Y = " << BeamWidthY() << " +/- " << BeamWidthYError() << " [cm]\n"
	   << " EmittanceX   = " << emittanceX() << " [cm]\n"
	   << " EmittanceY   = " << emittanceY() << " [cm]\n"
	   << " beta-star    = " << betaStar() << " [cm]\n"
       << "-----------------------------------------------------\n\n";
	
  }

  //
  std::ostream& operator<< ( std::ostream& os, BeamSpot beam ) {
    std::stringstream ss;
    beam.print(ss);
    os << ss.str();
    return os;
  }

  BeamSpot::Covariance3DMatrix BeamSpot::rotatedCovariance3D() const
  {
      AlgebraicVector3 newZ(dxdz(), dydz(), 1.);
      AlgebraicVector3 globalZ(0.,0.,1.);
      AlgebraicVector3 rotationAxis = ROOT::Math::Cross(globalZ.Unit(), newZ.Unit());
      float rotationAngle = -acos( ROOT::Math::Dot(globalZ.Unit(),newZ.Unit()));
      Basic3DVector<float> aa(rotationAxis[0],rotationAxis[1],rotationAxis[2]);
      TkRotation<float> rotation(aa ,rotationAngle);
      AlgebraicMatrix33 rotationMatrix;
      rotationMatrix(0,0) = rotation.xx();
      rotationMatrix(0,1) = rotation.xy();
      rotationMatrix(0,2) = rotation.xz();
      rotationMatrix(1,0) = rotation.yx();
      rotationMatrix(1,1) = rotation.yy();
      rotationMatrix(1,2) = rotation.yz();
      rotationMatrix(2,0) = rotation.zx();
      rotationMatrix(2,1) = rotation.zy();
      rotationMatrix(2,2) = rotation.zz();

      AlgebraicSymMatrix33 diagError ;
      diagError(0,0) = pow(BeamWidthX(),2);
      diagError(1,1) = pow(BeamWidthY(),2);
      diagError(2,2) = pow(sigmaZ(),2);

      Covariance3DMatrix matrix;
      matrix = ROOT::Math::Similarity(rotationMatrix, diagError) + covariance3D();
      return matrix;
  }

}
