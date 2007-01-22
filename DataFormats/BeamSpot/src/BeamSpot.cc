/**_________________________________________________________________
   class:   BeamSpot.cc
   package: DataFormats/BeamSpot
   
 A reconstructed beam spot providing position, width, slopes,
 and errors.

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpot.cc,v 1.1 2007/01/21 18:25:23 yumiceva Exp $

 ________________________________________________________________**/

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <iostream>


namespace reco {

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
}
