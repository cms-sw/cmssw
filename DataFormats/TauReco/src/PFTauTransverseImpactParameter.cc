#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TVectorT.h"
using namespace reco;

PFTauTransverseImpactParameter::PFTauTransverseImpactParameter(Point doca,double thedxy, double thedxy_error,VertexRef PV):
  doca_(doca),
  dxy_(thedxy),
  dxy_error_(thedxy_error),
  PV_(PV),
  hasSV_(false)
{
}

PFTauTransverseImpactParameter::PFTauTransverseImpactParameter(Point doca,double thedxy, double thedxy_error,VertexRef PV,Point theFlightLength,CovMatrix theFlightLenghtCov,VertexRef SV):
  doca_(doca),
  dxy_(thedxy),
  dxy_error_(thedxy_error),
  PV_(PV),
  hasSV_(true),
  FlightLength_(theFlightLength),
  FlightLenghtCov_(theFlightLenghtCov),
  SV_(SV)
{
}

PFTauTransverseImpactParameter* PFTauTransverseImpactParameter::clone() const{
  return new PFTauTransverseImpactParameter(*this);
}

double PFTauTransverseImpactParameter::FlightLengthSig(){
  if(hasSV_){
    VertexDistance3D vtxdist;
    return vtxdist.distance(*PV_, *SV_).significance(); // transforms using the jacobian then computes distance/uncertainty 
  }
  return -999;
}

TVector3 PFTauTransverseImpactParameter::FlightLength(){
  return TVector3(FlightLength_.x(),FlightLength_.y(),FlightLength_.z());
}
