#include "FastSimulation/ParamL3MuonProducer/interface/FML3PtSmearer.h" 

#include <cmath>

#include "FastSimulation/Utilities/interface/RandomEngine.h"

//
// Static data member definition
//
double FML3PtSmearer::MuonMassSquared_ = 0.10565837*0.10565837;


FML3PtSmearer::FML3PtSmearer(const RandomEngine * engine)
  : random(engine)
{ }

FML3PtSmearer::~FML3PtSmearer(){}

math::XYZTLorentzVector FML3PtSmearer::smear(math::XYZTLorentzVector simP4 , math::XYZVector recP3) const {
  double ptSim = std::sqrt(simP4.perp2());
  double invPtSim;
  if (ptSim>0.) invPtSim = 1./ptSim; 
  else { 
    // Better if we throw an exception here...
    simP4.SetPx(recP3.X());
    simP4.SetPy(recP3.Y());
    simP4.SetPz(recP3.Z());
    double muonEnergy=std::sqrt(simP4.P()*simP4.P()+MuonMassSquared_);
    simP4.SetE(muonEnergy);
    return simP4;
  }
  double etaSim = simP4.eta();
  double invPtNew = random->gaussShoot(invPtSim,error(ptSim,etaSim));
  invPtNew /= ( 1. + shift(ptSim,etaSim)*invPtNew);
  if (invPtNew>0.) {
    simP4.SetPx(simP4.x()*invPtSim/invPtNew);
    simP4.SetPy(simP4.y()*invPtSim/invPtNew);
    simP4.SetPz(recP3.z());
    double muonEnergy=std::sqrt(simP4.P()*simP4.P()+MuonMassSquared_);
    simP4.SetE(muonEnergy);
  }
  return simP4;
}

double FML3PtSmearer::error(double thePt, double theEta) const {
  return funSigma(fabs(theEta),thePt)/thePt;
}

double FML3PtSmearer::shift(double thePt, double theEta) const {
  return funShift(fabs(theEta));
}

double FML3PtSmearer::funShift(double x) const {
  if      (x<1.305) return 7.90897e-04;
  else if (x<1.82 ) return 9.52662e-02-1.12262e-01*x+3.05410e-02*x*x;
  else              return -7.9e-03;
  }

double FML3PtSmearer::funSigma(double eta , double pt) const {
  double sigma = funSigmaPt(pt) * funSigmaEta(eta);
  return sigma;
}

double FML3PtSmearer::funSigmaPt(double x) const {
  if (x<444.) return 3.13349e-01+2.77853e-03*x+4.94289e-06*x*x-9.63359e-09*x*x*x;
  else        return 9.26294e-01+1.64896e-03*x;
}

double FML3PtSmearer::funSigmaEta(double x) const {
  if (x<0.94) return 2.27603e-02+1.23995e-06*exp(9.30755*x);
  else        return 2.99467e-02+1.86770e-05*exp(3.52319*x);
}
