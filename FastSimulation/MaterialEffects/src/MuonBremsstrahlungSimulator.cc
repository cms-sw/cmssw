///////////////////////////////////////////////
//MuonBremsstrahlungSimulator
// Description: Implementation of Muon bremsstrahlung using the Petrukhin model
//Authors :Sandro Fonseca de Souza and Andre Sznajder (UERJ/Brazil)
// Date: 23-Nov-2010
//////////////////////////////////////////////////////////
#include "FastSimulation/MaterialEffects/interface/MuonBremsstrahlungSimulator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FastSimulation/MaterialEffects/interface/PetrukhinModel.h"

#include <cmath>
#include <string>
#include <iostream>

#include "TF1.h"




//////////////////////////////////////////////////////////////////////////////////////
MuonBremsstrahlungSimulator::MuonBremsstrahlungSimulator(const RandomEngine* engine,
							 double A,double Z, double density,double radLen,
							 double photonEnergyCut,double photonFractECut) : 
  MaterialEffectsSimulator(engine,A,Z,density,radLen) 
{
  // Set the minimal photon energy for a Brem from mu+/-
  photonEnergy = photonEnergyCut;
  photonFractE = photonFractECut;
  d = 0.; //distance
  LogDebug("MuonBremsstrahlungSimulator")<< "Starting the MuonBremsstrahlungSimulator"<< std::endl;
}



///////////////////////////////////////////////////////////////////
void 
MuonBremsstrahlungSimulator::compute(ParticlePropagator &Particle)
{

  double NA = 6.022e+23;  //Avogadro's number

  if ( radLengths > 4. )  {
    Particle.SetXYZT(0.,0.,0.,0.);
    deltaPMuon.SetXYZT(0.,0.,0.,0.);
    brem_photon.SetXYZT(0.,0.,0.,0.);
  }

// Hard brem probability with a photon Energy above photonEnergy.
  
  double EMuon = Particle.e();//Muon Energy
  if (EMuon<photonEnergy) return;
  xmin = std::max(photonEnergy/EMuon,photonFractE);//fraction of muon's energy transferred to the photon

 //  xmax = photonEnergy/Particle.e();
  if ( xmin >=1. || xmin <=0. ) return;
 
  xmax = 1.;
  npar = 3 ;//Number of parameters

  // create TF1 using a free C function 
  f1 = new TF1("f1",PetrukhinFunc,xmin,xmax,npar);
  //Setting parameters
  f1->SetParameters(EMuon,A,Z);
  ///////////////////////////////////////////////////////////////////////// 
  //d = distance for several materials
  //X0 = radLen
  //d = radLengths * X0(for tracker,yoke,ECAL and HCAL)
  d =  radLengths * radLen ;      
  //Integration
  bremProba = density * d *(NA/A)* (f1->Integral(0.,1.));
      
     
  // Number of photons to be radiated.
  unsigned int nPhotons = random->poissonShoot(bremProba);
  _theUpdatedState.reserve(nPhotons);
 
 
  if ( !nPhotons ) return;
 
  //Rotate to the lab frame
  double chi = Particle.theta();
  double psi = Particle.phi();
  RawParticle::RotationZ rotZ(psi);
  RawParticle::RotationY rotY(chi);

 
 
  // Energy of these photons
  for ( unsigned int i=0; i<nPhotons; ++i ) {

     // Check that there is enough energy left.
    if ( Particle.e() < photonEnergy ) break;
    LogDebug("MuonBremsstrahlungSimulator")<< "MuonBremsstrahlungSimulator parameters:"<< std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "xmin-> " << xmin << std::endl; 
    LogDebug("MuonBremsstrahlungSimulator")<< "Atomic Weight-> " << A << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "Density-> " << density << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "Distance-> " << d << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "bremProba->" << bremProba << std::endl;    
    LogDebug("MuonBremsstrahlungSimulator")<< "nPhotons->" << nPhotons << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< " Muon_Energy-> " <<  EMuon << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "X0-> "<< radLen << std::endl; 
    LogDebug("MuonBremsstrahlungSimulator")<< " radLengths-> " << radLengths << std::endl; 

    // Add a photon
    RawParticle thePhoton(22,brem(Particle));    
    if (thePhoton.E()>0.){

    thePhoton.rotate(rotY);
    thePhoton.rotate(rotZ);
    
    _theUpdatedState.push_back(thePhoton);
	
    // Update the original mu +/-
    deltaPMuon = Particle -= thePhoton.momentum();
    // Information of brem photon
    brem_photon.SetXYZT(thePhoton.Px(),thePhoton.Py(),thePhoton.Pz(),thePhoton.E());     

    LogDebug("MuonBremsstrahlungSimulator")<< " Muon Bremsstrahlung: photon_energy-> " << thePhoton.E() << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "photon_px->" << thePhoton.Px() << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "photon_py->" << thePhoton.Py() << std::endl;
    LogDebug("MuonBremsstrahlungSimulator")<< "photon_pz->" << thePhoton.Pz() << std::endl;
    
    } 
    
  }

}
///////////////////////////////////////////////////////////////////////////////////////////
XYZTLorentzVector 
MuonBremsstrahlungSimulator::brem(ParticlePropagator& pp) const {

  // This is a simple version of a Muon Brem using Petrukhin model .
  //Ref: http://pdg.lbl.gov/2008/AtomicNuclearProperties/adndt.pdf 
  double mumass = 0.105658367;//mu mass  (GeV/c^2)
  double xp = f1->GetRandom();  
  LogDebug("MuonBremsstrahlungSimulator")<<  "MuonBremsstrahlungSimulator: xp->" << xp << std::endl;
  std::cout << "MuonBremsstrahlungSimulator: xp->" << xp << std::endl;

  
  // Have photon energy. Now generate angles with respect to the z axis 
  // defined by the incoming particle's momentum.

  // Isotropic in phi
  const double phi = random->flatShoot()*2*M_PI;
  // theta from universal distribution
  const double theta = gbteth(pp.e(),mumass,xp)*mumass/pp.e(); 
  
  // Make momentum components
  double stheta = std::sin(theta);
  double ctheta = std::cos(theta);
  double sphi   = std::sin(phi);
  double cphi   = std::cos(phi);

  return xp * pp.e() * XYZTLorentzVector(stheta*cphi,stheta*sphi,ctheta,1.);
 
}
//////////////////////////////////////////////////////////////////////////////////////////////
double
MuonBremsstrahlungSimulator::gbteth(const double ener,
				const double partm,
				const double efrac) const {
  const double alfa = 0.625;

  const double d = 0.13*(0.8+1.3/theZ())*(100.0+(1.0/ener))*(1.0+efrac);
  const double w1 = 9.0/(9.0+d);
  const double umax = ener*M_PI/partm;
  double u;
  
  do {
    double beta = (random->flatShoot()<=w1) ? alfa : 3.0*alfa;
    u = -std::log(random->flatShoot()*random->flatShoot())/beta;
  } while (u>=umax);

  return u;
}


