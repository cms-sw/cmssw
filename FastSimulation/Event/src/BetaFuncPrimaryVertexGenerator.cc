//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Famos Headers
#include "FastSimulation/Event/interface/BetaFuncPrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

BetaFuncPrimaryVertexGenerator::BetaFuncPrimaryVertexGenerator(
  const edm::ParameterSet& vtx, const RandomEngine* engine) : 
  PrimaryVertexGenerator(engine),
  fX0(vtx.getParameter<double>("X0")),
  fY0(vtx.getParameter<double>("Y0")),
  fZ0(vtx.getParameter<double>("Z0")),
  fSigmaZ(vtx.getParameter<double>("SigmaZ")),
  alpha_(vtx.getParameter<double>("Alpha")),
  phi_(vtx.getParameter<double>("Phi")),
  fbetastar(vtx.getParameter<double>("BetaStar")),
  femittance(vtx.getParameter<double>("Emittance"))
{

  this->setBoost(inverseLorentzBoost());
  beamSpot_ = math::XYZPoint(fX0,fY0,fZ0);

} 
  

void BetaFuncPrimaryVertexGenerator::generate() {	
  
  double tmp_sigz = random->gaussShoot(0., fSigmaZ);
  this->SetZ(tmp_sigz + fZ0);

  double tmp_sigx = BetaFunction(tmp_sigz,fZ0); 
  // need to divide by sqrt(2) for beamspot width relative to single beam width
  tmp_sigx *= 0.707107;
  this->SetX(random->gaussShoot(fX0,tmp_sigx));

  double tmp_sigy = BetaFunction(tmp_sigz,fZ0);
  // need to divide by sqrt(2) for beamspot width relative to single beam width
  tmp_sigy *= 0.707107;
  this->SetY(random->gaussShoot(fY0,tmp_sigy));

}

double BetaFuncPrimaryVertexGenerator::BetaFunction(double z, double z0)
{
  return sqrt(femittance*(fbetastar+(((z-z0)*(z-z0))/fbetastar)));

}


TMatrixD* 
BetaFuncPrimaryVertexGenerator::inverseLorentzBoost() {
  
  TMatrixD* aBoost = 0;
  if ( fabs(alpha_) < 1E-12 && fabs(phi_) < 1E-12 ) return aBoost;
 
  TMatrixD tmpboost(4,4);
  
  // Lorentz boost to frame where the collision is head-on
  // phi is the half crossing angle in the plane ZS
  // alpha is the angle to the S axis from the X axis in the XY plane

  double calpha = std::cos(alpha_);
  double salpha = std::sin(alpha_);
  double cphi = std::cos(phi_);
  double sphi = std::sin(phi_);
  double tphi = sphi/cphi;
  tmpboost(0,0) = 1./cphi;
  tmpboost(0,1) = - calpha*sphi;
  tmpboost(0,2) = - tphi*sphi;
  tmpboost(0,3) = - salpha*sphi;
  tmpboost(1,0) = - calpha*tphi;
  tmpboost(1,1) = 1.;
  tmpboost(1,2) = calpha*tphi;
  tmpboost(1,3) = 0.;
  tmpboost(2,0) = 0.;
  tmpboost(2,1) = -calpha*sphi;
  tmpboost(2,2) = cphi;
  tmpboost(2,3) = - salpha*sphi;
  tmpboost(3,0) = - salpha*tphi;
  tmpboost(3,1) = 0.;
  tmpboost(3,2) = salpha*tphi;
  tmpboost(3,3) = 1.;
  
  tmpboost.Invert();
  aBoost = new TMatrixD(tmpboost);

  return aBoost;

}
