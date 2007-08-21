/**_________________________________________________________________
   class:   BeamSpot.cc
   package: DataFormats/BeamSpot
   
 A reconstructed beam spot providing position, width, slopes,
 and errors.

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpot.cc,v 1.3 2007/06/27 12:25:48 speer Exp $

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
    BeamWidth_ = 0.;

  }

  void BeamSpot::dummy() {
    // dummy beam spot
    position_ = Point(0.,0.,0.);
    sigmaZ_ = 7.55; //cm
    dxdz_ = 0.;
    dydz_ = 0.;
    BeamWidth_ = 0.0015; //cm
    error_(0,0) = BeamWidth_*BeamWidth_;
    error_(1,1) = error_(0,0);
    error_(2,2) = sigmaZ_*sigmaZ_;

  }

  void BeamSpot::print(std::stringstream& ss) const {

    ss << "-----------------------------------------------------\n"
       << "            Calculated Beam Spot\n\n"
       << "   X0 = " << x0() << " +/- " << x0Error() << " [cm]\n"
       << "   Y0 = " << y0() << " +/- " << y0Error() << " [cm]\n"
       << "   Z0 = " << z0() << " +/- " << z0Error() << " [cm]\n"
       << " Sigma Z0 = " << sigmaZ() << " +/- " << sigmaZ0Error() << " [cm]\n"
       << " dxdz = " << dxdz() << " +/- " << dxdzError() << " [radians]\n"
       << " dydz = " << dydz() << " +/- " << dydzError() << " [radians]\n"
       << " Beam Width = " << BeamWidth() << " +/- " << BeamWidthError() << " [cm]\n"
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
      AlgebraicVector a = asHepVector(rotationAxis);
      Basic3DVector<float> aa(a[0], a[1], a[2]);
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

      AlgebraicSymMatrix33 diagError = covariance3D();
      diagError(0,0) += pow(BeamWidth(),2);
      diagError(1,1) += pow(BeamWidth(),2);
      diagError(2,2) += pow(sigmaZ(),2);

      Covariance3DMatrix matrix;
      matrix = ROOT::Math::Similarity(rotationMatrix, diagError);
      return matrix;
  }

}
