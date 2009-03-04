#include "GeneratorInterface/CosmicMuonGenerator/interface/SingleParticleEvent.h"

void SingleParticleEvent::create(int id, double px, double py, double pz, double e, double m, double vx, double vy, double vz, double t0){
    ID = id;
    Px = px; Py = py; Pz = pz; E = e; M = m;
    Vx = vx; Vy = vy; Vz = vz; T0 = t0;
    HitTarget = false;
}

void SingleParticleEvent::propagate(double ElossScaleFac, double RadiusTarget, double Z_DistTarget, double Z_CentrTarget, bool TrackerOnly, bool MTCCHalf){
  MTCC=MTCCHalf; //need to know this boolean in absVzTmp()
  // calculated propagation direction
  dX = Px/absmom();
  dY = Py/absmom(); 
  dZ = Pz/absmom();
  // propagate with decreasing step size
  tmpVx = Vx;
  tmpVy = Vy;
  tmpVz = Vz;
  double RadiusTargetEff = RadiusTarget;
  double Z_DistTargetEff = Z_DistTarget;
  double Z_CentrTargetEff = Z_CentrTarget;
  if(TrackerOnly==true){
    RadiusTargetEff = RadiusTracker;
    Z_DistTargetEff = Z_DistTracker;
  }
  HitTarget = true;
  if (HitTarget == true){
    HitTarget = false;
    double stepSize = MinStepSize*100000.;
    double acceptR = RadiusTargetEff + stepSize;
    double acceptZ = Z_DistTargetEff + stepSize;
    bool continuePropagation = true;
    while (continuePropagation){
      //if (tmpVy < -acceptR) continuePropagation = false;
      if (dY < 0. && tmpVy < -acceptR) continuePropagation = false;
      if (dY >= 0. && tmpVy > acceptR) continuePropagation = false;
      //if (absVzTmp() < acceptZ && rVxyTmp() < acceptR){
      if (fabs(tmpVz - Z_CentrTargetEff) < acceptZ && rVxyTmp() < acceptR){
	HitTarget = true;
	continuePropagation = false;
      }
      if (continuePropagation) updateTmp(stepSize);
    }
  }
  if (HitTarget == true){
    HitTarget = false;
    double stepSize = MinStepSize*10000.;
    double acceptR = RadiusTargetEff + stepSize;
    double acceptZ = Z_DistTargetEff + stepSize;
    bool continuePropagation = true;
    while (continuePropagation){
      //if (tmpVy < -acceptR) continuePropagation = false;
      if (dY < 0. && tmpVy < -acceptR) continuePropagation = false;
      if (dY >= 0. && tmpVy > acceptR) continuePropagation = false;
      //if (absVzTmp() < acceptZ && rVxyTmp() < acceptR){
      if (fabs(tmpVz - Z_CentrTargetEff) < acceptZ && rVxyTmp() < acceptR){
	HitTarget = true;
	continuePropagation = false;
      }
      if (continuePropagation) updateTmp(stepSize);
    }
  }
  if (HitTarget == true){
    HitTarget = false;
    double stepSize = MinStepSize*1000.;
    double acceptR = RadiusTargetEff + stepSize;
    double acceptZ = Z_DistTargetEff + stepSize;
    bool continuePropagation = true;
    while (continuePropagation){
      //if (tmpVy < -acceptR) continuePropagation = false;
      if (dY < 0. && tmpVy < -acceptR) continuePropagation = false;
      if (dY >= 0. && tmpVy > acceptR) continuePropagation = false;
      //if (absVzTmp() < acceptZ && rVxyTmp() < acceptR){
      if (fabs(tmpVz - Z_CentrTargetEff) < acceptZ && rVxyTmp() < acceptR){
	HitTarget = true;
	continuePropagation = false;
      }
      if (continuePropagation) updateTmp(stepSize);
    }
  }
  if (HitTarget == true){
    HitTarget = false;
    double stepSize = MinStepSize*100.;
    double acceptR = RadiusTargetEff + stepSize;
    double acceptZ = Z_DistTargetEff + stepSize;
    bool continuePropagation = true;
    while (continuePropagation){
      //if (tmpVy < -acceptR) continuePropagation = false;
      if (dY < 0. && tmpVy < -acceptR) continuePropagation = false;
      if (dY >= 0. && tmpVy > acceptR) continuePropagation = false;
      //if (absVzTmp() < acceptZ && rVxyTmp() < acceptR){
      if (fabs(tmpVz - Z_CentrTargetEff) < acceptZ && rVxyTmp() < acceptR){
	HitTarget = true;
	continuePropagation = false;
      }
      if (continuePropagation) updateTmp(stepSize);
    }
  }
  if (HitTarget == true){
    HitTarget = false;
    double stepSize = MinStepSize*10.;
    double acceptR = RadiusTargetEff + stepSize;
    double acceptZ = Z_DistTargetEff + stepSize;
    bool continuePropagation = true;
    while (continuePropagation){
      //if (tmpVy < -acceptR) continuePropagation = false;
      if (dY < 0. && tmpVy < -acceptR) continuePropagation = false;
      if (dY >= 0. && tmpVy > acceptR) continuePropagation = false;
      //if (absVzTmp() < acceptZ && rVxyTmp() < acceptR){
      if (fabs(tmpVz - Z_CentrTargetEff) < acceptZ && rVxyTmp() < acceptR){
	HitTarget = true;
	continuePropagation = false;
      }
      if (continuePropagation) updateTmp(stepSize);
    }
  }
  if (HitTarget == true){
    HitTarget = false;
    double stepSize = MinStepSize*1.;
    double acceptR = RadiusTargetEff + stepSize;
    double acceptZ = Z_DistTargetEff + stepSize;
    bool continuePropagation = true;
    while (continuePropagation){
      //if (tmpVy < -acceptR) continuePropagation = false;
      if (dY < 0. && tmpVy < -acceptR) continuePropagation = false;
      if (dY >= 0. && tmpVy > acceptR) continuePropagation = false;
      if (0 < absVzTmp()){ //only check for MTCC setup in last step of propagation, need fine stepSize
	//if (absVzTmp() < acceptZ && rVxyTmp() < acceptR){
	if (fabs(tmpVz - Z_CentrTargetEff) < acceptZ && rVxyTmp() < acceptR){
	  HitTarget = true;
	  continuePropagation = false;
	}
      }
      if (continuePropagation) updateTmp(stepSize);
    }
  }
  // actual propagation + energy loss
  if (HitTarget == true){
    HitTarget = false;
    //int nAir = 0; int nWall = 0; int nRock = 0; int nClay = 0; int nPlug = 0;
    int nMat[6] = {0, 0, 0, 0, 0, 0};
    double stepSize = MinStepSize*1.; // actual step size
    double acceptR = RadiusCMS + stepSize;
    double acceptZ = Z_DistCMS + stepSize;
    if(TrackerOnly==true){
      acceptR = RadiusTracker + stepSize;
      acceptZ = Z_DistTracker + stepSize;
    }
    bool continuePropagation = true;
    while (continuePropagation){
      //if (Vy < -acceptR) continuePropagation = false;
      if (dY < 0. && tmpVy < -acceptR) continuePropagation = false;
      if (dY >= 0. && tmpVy > acceptR) continuePropagation = false;
      //if (absVz() < acceptZ && rVxy() < acceptR){
      if (fabs(Vz - Z_CentrTargetEff) < acceptZ && rVxy() < acceptR){
        HitTarget = true;
        continuePropagation = false;
      }
      if (continuePropagation) update(stepSize);

      int Mat = inMat(Vx,Vy,Vz, PlugVx, PlugVz);

      nMat[Mat]++;
    }

    if (HitTarget){
      double lPlug = double(nMat[Plug])*stepSize;
      double lWall = double(nMat[Wall])*stepSize;
      double lAir = double(nMat[Air])*stepSize;
      double lClay = double(nMat[Clay])*stepSize;
      double lRock = double(nMat[Rock])*stepSize;      
      //double lUnknown = double(nMat[Unknown])*stepSize;

      double waterEquivalents = (lAir*RhoAir + lWall*RhoWall + lRock*RhoRock
				 + lClay*RhoClay + lPlug*RhoPlug) *ElossScaleFac/10.; // [g cm^-2]
      subtractEloss(waterEquivalents);
      if (E < MuonMass) HitTarget = false; // muon stopped in the material around the target
    }
  }
  // end of propagation part
}

void SingleParticleEvent::update(double stepSize){
  Vx += stepSize*dX;
  Vy += stepSize*dY;
  Vz += stepSize*dZ;
}

void SingleParticleEvent::updateTmp(double stepSize){
  tmpVx += stepSize*dX;
  tmpVy += stepSize*dY;
  tmpVz += stepSize*dZ;
}

void SingleParticleEvent::subtractEloss(double waterEquivalents){
  double L10E = log10(E);
  // parameters for standard rock (PDG 2004, page 230)
  double A = (1.91514 + 0.254957*L10E)/1000.;                         // a [GeV g^-1 cm^2]
  double B = (0.379763 + 1.69516*L10E - 0.175026*L10E*L10E)/1000000.; // b [g^-1 cm^2]
  double EPS = A/B;                                                   // epsilon [GeV]
  E = (E + EPS)*exp(-B*waterEquivalents) - EPS; // updated energy
  double oldAbsMom = absmom();
  double newAbsMom = sqrt(E*E - MuonMass*MuonMass);
  Px = Px*newAbsMom/oldAbsMom;                  // updated px
  Py = Py*newAbsMom/oldAbsMom;                  // updated py
  Pz = Pz*newAbsMom/oldAbsMom;                  // updated pz
}

double SingleParticleEvent::absVzTmp(){
  if(MTCC==true){
    return tmpVz; //need sign to be sure muon hits half of CMS with MTCC setup
  }else{
    return fabs(tmpVz);
  }
}

double SingleParticleEvent::rVxyTmp(){
  return sqrt(tmpVx*tmpVx + tmpVy*tmpVy);
}

bool SingleParticleEvent::hitTarget(){ return HitTarget; }

int    SingleParticleEvent::id(){ return ID; }

double SingleParticleEvent::px(){ return Px; }

double SingleParticleEvent::py(){ return Py; }

double SingleParticleEvent::pz(){ return Pz; }

double SingleParticleEvent::e(){ return E; }

double SingleParticleEvent::m(){ return M; }

double SingleParticleEvent::vx(){ return Vx; }

double SingleParticleEvent::vy(){ return Vy; }

double SingleParticleEvent::vz(){ return Vz; }

double SingleParticleEvent::t0(){ return T0; }

double SingleParticleEvent::phi(){
  double phiXZ = atan2(Px,Pz);
  if (phiXZ < 0.) phiXZ = phiXZ + TwoPi;
  return  phiXZ;
}

double SingleParticleEvent::theta(){
  return atan2(sqrt(Px*Px+Pz*Pz),-Py);
}

double SingleParticleEvent::absmom(){
  return sqrt(Px*Px + Py*Py + Pz*Pz);
}

double SingleParticleEvent::absVz(){
  return fabs(Vz);
}

double SingleParticleEvent::rVxy(){
  return sqrt(Vx*Vx + Vy*Vy);
}
