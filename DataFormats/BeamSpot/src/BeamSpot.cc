/**_________________________________________________________________
   class:   BeamSpot.cc
   package: DataFormats/BeamSpot
   
 A reconstructed beam spot providing position, width, slopes,
 and errors.

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpot.cc,v 1.1 2006/12/15 20:00:37 yumiceva Exp $

 ________________________________________________________________**/

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

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

}
