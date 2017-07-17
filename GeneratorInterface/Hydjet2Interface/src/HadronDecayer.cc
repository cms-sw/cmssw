/*

July 2008 BW mass is limited by "PYTHIA method", by I. Lokhtin and L. Malinina
                                                                          
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#include <functional>
#include <algorithm>
#include <vector>
#include <iostream>
#include <TMath.h>

#include "GeneratorInterface/Hydjet2Interface/interface/DatabasePDG.h"
#include "GeneratorInterface/Hydjet2Interface/interface/ParticlePDG.h"
#include "GeneratorInterface/Hydjet2Interface/interface/DecayChannel.h"
#include "GeneratorInterface/Hydjet2Interface/interface/HadronDecayer.h"
#include "GeneratorInterface/Hydjet2Interface/interface/UKUtility.h"
#include "GeneratorInterface/Hydjet2Interface/interface/Particle.h"
#include "GeneratorInterface/Hydjet2Interface/interface/HYJET_COMMONS.h"

//calculates decay time in fm/c
//calculates 1,2 and 3 body decays

using namespace std;

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandBreitWigner.h"
extern CLHEP::HepRandomEngine* hjRandomEngine;


double GetDecayTime(const Particle &parent, double weakDecayLimit) {
  ParticlePDG *pDef = parent.Def(); 
  double width = pDef->GetWidth(); //GeV

  double fullBranching = pDef->GetFullBranching(); // Only 3 or less body decays
  // check if full branching is 100%
  if(pDef->GetNDecayChannels()>0 && fullBranching<0.9999) {  
    LogDebug("HadronDecayer") << "GetDecayTime(): WARNING!!! The full branching for specie " << pDef->GetPDG() << " is less than 100% (" 
    	 << fullBranching*100 << "%) and decay channels exist!";
  }

  if(width > weakDecayLimit) {
    double slope =  parent.E() * 0.1973 / (pDef->GetMass() * width);
    return -slope * TMath::Log(CLHEP::RandFlat::shoot(hjRandomEngine));//in fm/c
  }

  return 0.;
}

extern "C" void mydelta_();
extern SERVICEEVCommon SERVICEEV;

void Decay(List_t &output, Particle &parent, ParticleAllocator &allocator, DatabasePDG* database) {
  // Get the PDG properties of the particle
  ParticlePDG *pDef = parent.Def();

  // Get the number of posible decay channels
  int nDecayChannel = pDef->GetNDecayChannels();
  if(pDef->GetWidth()>0 && nDecayChannel==0) {
    LogDebug("HadronDecayer") << "Decay(): WARNING!! Particle " << pDef->GetPDG() << " has finite width (" << pDef->GetWidth()
         	 << ") but NO decay channels specified in the database. Check it out!!";
    return;
  }

  // check the full branching of this specie
  double fullBranching = pDef->GetFullBranching(); // Only 3 or less body decays

  // return if particle has no branching
  if(fullBranching < 0.00001)
    return;

  // get the PDG mass of the specie  
  double PDGmass = pDef->GetMass();
  int ComprCodePyth=0;
  float Delta =0;

  bool success = kFALSE;
  int iterations = 0;
  // Try to decay the particle
  while(!success) {
    if(iterations>1000) { //???
      LogDebug("HadronDecayer") << "Decay(): WARNING!!! iterations to decay "
      	   << pDef->GetPDG() << " : " << iterations << "   Check it out!!!";
    }

    // get a random mass using the Breit-Wigner distribution
    double BWmass = CLHEP::RandBreitWigner::shoot(hjRandomEngine, PDGmass, pDef->GetWidth());

    // Try to cut the Breit Wigner tail of the particle using the cuts from pythia
    // The Delta variable is obtained from pythia based on the specie
    int encoding = pDef->GetPDG();
    SERVICEEV.ipdg = encoding;
    mydelta_();      
    ComprCodePyth=SERVICEEV.KC;
    Delta = SERVICEEV.delta;// PYDAT2.PMAS[KC][3];
    
    //if there are no such particle in PYTHIA particle table, we take Delta=0.4
    if(ComprCodePyth==0){
      BWmass=PDGmass; 
      Delta=0.0;
    } 

    //bad delta - an exception
    if(ComprCodePyth==254){
      BWmass=PDGmass; 
      Delta=0.0;
    } 

    // K0 decay into K0s or K0l
    if(TMath::Abs(encoding)==311) {
      BWmass=PDGmass;
      Delta=0.0;
    }
      
    //for particles from PYTHIA table only, if the BW mass is outside the cut range then quit this iteration and generate another BW mass
    if(ComprCodePyth!=0 && Delta>0 && (BWmass<PDGmass-Delta || BWmass>PDGmass+Delta)){
      iterations++;
      continue;
    }    
        
        if(BWmass>5)
          LogDebug("HadronDecayer") << "Decay(): Breit-Wigner mass > 5GeV for encoding: "<< encoding
    	       <<"; PDG mass: "<< PDGmass <<"; delta: " << Delta << "; width: "
    	       << pDef->GetWidth() << "; mass: " << BWmass << "; ComprCodePyth: " << ComprCodePyth;
    
    // check how many decay channels are allowed with the generated mass
    int nAllowedChannels = database->GetNAllowedChannels(pDef, BWmass);
    // if no decay channels are posible with this mass, then generate another BW mass
    if(nAllowedChannels==0) {
      iterations++;
      continue;
    }

    std::vector<Particle> apDaughter;
    std::vector<double> dMass; //daughters'mass
    std::vector<double> dMom;
    std::vector<double> sm;
    std::vector<double> rd;

    // we need to choose an allowed decay channel
    double randValue = CLHEP::RandFlat::shoot(hjRandomEngine) * fullBranching;

    int chosenChannel = 1000;
    bool found = kFALSE;
    int channelIterations = 0;
    while(!found) {
      if(channelIterations > 1000) {
        LogDebug("HadronDecayer") << "Decay(): More than 1000 iterations to choose a decay channel. Check it out !!";
      }
      for(int nChannel = 0; nChannel < nDecayChannel; ++nChannel) {
	randValue -= pDef->GetDecayChannel(nChannel)->GetBranching();
	if(randValue <= 0. && database->IsChannelAllowed(pDef->GetDecayChannel(nChannel), BWmass)) {
	  chosenChannel = nChannel;
	  found = kTRUE;
	  break;
	}
      }
      channelIterations++;
    }

    // get the PDG information for the chosen decay channel
    DecayChannel *dc = pDef->GetDecayChannel(chosenChannel);
    int nSec = dc->GetNDaughters();

    // Adjust the parent momentum four-vector for the MC generated Breit-Wigner mass
    Particle parentBW(database->GetPDGParticle(parent.Encoding()));
    parentBW.Pos(parent.Pos());
    double BWenergy = TMath::Sqrt(parent.Mom().X()*parent.Mom().X() + 
                                  parent.Mom().Y()*parent.Mom().Y() +
                                  parent.Mom().Z()*parent.Mom().Z() +
                                  BWmass*BWmass);

    int NB = (int)parent.GetType(); //particle from jets

    TLorentzVector MomparentBW(parent.Mom().X(), parent.Mom().Y(), parent.Mom().Z(), BWenergy); 
    parentBW.Mom(MomparentBW);
    // take into account BW when calculating boost velocity (for wide resonances it matters)
    TVector3 velocityBW(parentBW.Mom().BoostVector());

    // now we have an allowed decay
    // first case: one daughter particle
    if(nSec == 1) {
      // initialize the daughter particle
      Particle p1(database->GetPDGParticle(dc->GetDaughterPDG(0)));
      p1.Pos(parentBW.Pos());
      p1.Mom(parent.Mom());
      p1.SetLastMotherPdg(parentBW.Encoding());
      p1.SetLastMotherDecayCoor(parentBW.Pos());
      p1.SetLastMotherDecayMom(parentBW.Mom());
      p1.SetType(NB);
      p1.SetPythiaStatusCode(parent.GetPythiaStatusCode());
      
      // add the daughter particle to the list of secondaries
      int parentIndex = parent.GetIndex(); 
      int p1Index = p1.SetIndex();
      p1.SetMother(parentIndex);
      parent.SetFirstDaughterIndex(p1Index);
      parent.SetLastDaughterIndex(p1Index);
      allocator.AddParticle(p1, output);
      success = kTRUE;  
    }
    // second case: two daughter particles
    else if(nSec == 2) {
      // initialize the daughter particles
      Particle p1(database->GetPDGParticle(dc->GetDaughterPDG(0)));
      p1.Pos(parentBW.Pos());
      Particle p2(database->GetPDGParticle(dc->GetDaughterPDG(1)));
      p2.Pos(parentBW.Pos());
      
      // calculate the momenta in rest frame of mother for the two particles (theta and phi are isotropic)
      MomAntiMom(p1.Mom(), p1.TableMass(), p2.Mom(), p2.TableMass(), BWmass);
    
      // boost to the laboratory system (to the mother velocity)
      p1.Mom().Boost(velocityBW);
      p2.Mom().Boost(velocityBW);

      //store information about mother
      p1.SetLastMotherPdg(parentBW.Encoding());
      p1.SetLastMotherDecayCoor(parentBW.Pos());
      p1.SetLastMotherDecayMom(parentBW.Mom());
      p2.SetLastMotherPdg(parentBW.Encoding());
      p2.SetLastMotherDecayCoor(parentBW.Pos());
      p2.SetLastMotherDecayMom(parentBW.Mom());
      //set to daughters the same type as has mother
      p1.SetType(NB);
      p2.SetType(NB);
      p1.SetPythiaStatusCode(parent.GetPythiaStatusCode());
      p2.SetPythiaStatusCode(parent.GetPythiaStatusCode());

      // check the kinematics in the lab system
      double deltaS = TMath::Sqrt((parentBW.Mom().X()-p1.Mom().X()-p2.Mom().X())*(parentBW.Mom().X()-p1.Mom().X()-p2.Mom().X())+
                                  (parentBW.Mom().Y()-p1.Mom().Y()-p2.Mom().Y())*(parentBW.Mom().Y()-p1.Mom().Y()-p2.Mom().Y())+
                                  (parentBW.Mom().Z()-p1.Mom().Z()-p2.Mom().Z())*(parentBW.Mom().Z()-p1.Mom().Z()-p2.Mom().Z())+
                                  (parentBW.Mom().E()-p1.Mom().E()-p2.Mom().E())*(parentBW.Mom().E()-p1.Mom().E()-p2.Mom().E()));
      // if deltaS is too big then repeat the kinematic procedure
 
 
      if(deltaS>0.001) {
        
          LogDebug("HadronDecayer|deltaS") << "2-body decay kinematic check in lab system: " << pDef->GetPDG() << " >>> " << p1.Encoding() << " + " << p2.Encoding() << endl
          << " Mother    (e,px,py,pz): " << parentBW.Mom().E() << " : " << parentBW.Mom().X() << " : " << parentBW.Mom().Y() << " : " << parentBW.Mom().Z() << endl
          << " Mother    (x,y,z,t): " << parentBW.Pos().X() << " : " << parentBW.Pos().Y() << " : " << parentBW.Pos().Z() << " : " << parentBW.Pos().T() << endl
          << " Daughter1 (e,px,py,pz): " << p1.Mom().E() << " : " << p1.Mom().X() << " : " << p1.Mom().Y() << " : " << p1.Mom().Z() << endl
          << " Daughter2 (e,px,py,pz): " << p2.Mom().E() << " : " << p2.Mom().X() << " : " << p2.Mom().Y() << " : " << p2.Mom().Z() << endl	
          << " 2-body decay delta(sqrtS): " << deltaS << endl
          << " Repeating the decay algorithm";
        
	iterations++;
	continue;
      }
      // push particles to the list of secondaries
      int parentIndex = parent.GetIndex();
      p1.SetIndex(); 
      p2.SetIndex();
      p1.SetMother(parentIndex); 
      p2.SetMother(parentIndex);
      parent.SetFirstDaughterIndex(p1.GetIndex());
      parent.SetLastDaughterIndex(p2.GetIndex());
      allocator.AddParticle(p1, output);
      allocator.AddParticle(p2, output);
      success = kTRUE;
    }

    // third case: three daughter particle
    else if(nSec == 3) {
      // initialize the daughter particle
      Particle p1(database->GetPDGParticle(dc->GetDaughterPDG(0)));
      p1.Pos(parentBW.Pos());
      Particle p2(database->GetPDGParticle(dc->GetDaughterPDG(1)));
      p2.Pos(parentBW.Pos());
      Particle p3(database->GetPDGParticle(dc->GetDaughterPDG(2)));
      p3.Pos(parentBW.Pos());
      // calculate the momenta in the rest frame of the mother particle
      double pAbs1 = 0., pAbs2 = 0., pAbs3 = 0., sumPabs = 0., maxPabs = 0.;
      double mass1 = p1.TableMass(), mass2 = p2.TableMass(), mass3 = p3.TableMass();
      TLorentzVector &mom1 = p1.Mom(), &mom2 = p2.Mom(), &mom3 = p3.Mom(); 
      double deltaMass = BWmass - mass1 - mass2 - mass3;

      do {
	double rd1 = CLHEP::RandFlat::shoot(hjRandomEngine);
	double rd2 = CLHEP::RandFlat::shoot(hjRandomEngine);
	if (rd2 > rd1)
	  std::swap(rd1, rd2);
	// 1
	double e = rd2*deltaMass;
	pAbs1 = TMath::Sqrt(e*e + 2*e*mass1);
	sumPabs = pAbs1;
	maxPabs = sumPabs;
	// 2
	e = (1-rd1)*deltaMass;
	pAbs2 = TMath::Sqrt(e*e + 2*e*mass2);
	
	if(pAbs2 > maxPabs)
	  maxPabs = pAbs2;
	
	sumPabs += pAbs2;
	// 3
	e = (rd1-rd2)*deltaMass;
	pAbs3 = TMath::Sqrt(e*e + 2*e*mass3);
	
	if (pAbs3 > maxPabs)
	  maxPabs =  pAbs3;
	sumPabs  +=  pAbs3;
      } while(maxPabs > sumPabs - maxPabs);
      
      // isotropic sample first particle 3-momentum
      double cosTheta = 2*(CLHEP::RandFlat::shoot(hjRandomEngine)) - 1;
      double sinTheta = TMath::Sqrt(1 - cosTheta*cosTheta);
      double phi      = TMath::TwoPi()*(CLHEP::RandFlat::shoot(hjRandomEngine));
      double sinPhi   = TMath::Sin(phi);
      double cosPhi   = TMath::Cos(phi);
      
      mom1.SetPxPyPzE(sinTheta*cosPhi, sinTheta*sinPhi, cosTheta, 0);
      mom1 *= pAbs1;
      // sample rest particle 3-momentum
      double cosThetaN = (pAbs2*pAbs2 - pAbs3*pAbs3 - pAbs1*pAbs1)/(2*pAbs1*pAbs3);
      double sinThetaN = TMath::Sqrt(1 - cosThetaN*cosThetaN);
      double phiN      = TMath::TwoPi()*(CLHEP::RandFlat::shoot(hjRandomEngine));
      double sinPhiN   = TMath::Sin(phiN);
      double cosPhiN   = TMath::Cos(phiN);
      
      mom3.SetPxPyPzE(sinThetaN*cosPhiN*cosTheta*cosPhi - sinThetaN*sinPhiN*sinPhi + cosThetaN*sinTheta*cosPhi,
		      sinThetaN*cosPhiN*cosTheta*sinPhi + sinThetaN*sinPhiN*cosPhi + cosThetaN*sinTheta*sinPhi,
		      -sinThetaN*cosPhiN*sinTheta + cosThetaN*cosTheta,
		      0.);
      
      mom3 *= pAbs3*mom3.P();
      mom2 = mom1;
      mom2 += mom3;
      mom2 *= -1.;
      // calculate energy
      mom1.SetE(TMath::Sqrt(mom1.P()*mom1.P() + mass1*mass1));
      mom2.SetE(TMath::Sqrt(mom2.P()*mom2.P() + mass2*mass2));
      mom3.SetE(TMath::Sqrt(mom3.P()*mom3.P() + mass3*mass3));
      
      // boost to Lab system
      mom1.Boost(velocityBW);
      mom2.Boost(velocityBW);
      mom3.Boost(velocityBW);
      
      p1.SetLastMotherPdg(parentBW.Encoding());
      p1.SetLastMotherDecayCoor(parentBW.Pos());
      p1.SetLastMotherDecayMom(parentBW.Mom());
      p2.SetLastMotherPdg(parentBW.Encoding());
      p2.SetLastMotherDecayCoor(parentBW.Pos());
      p2.SetLastMotherDecayMom(parentBW.Mom());
      p3.SetLastMotherPdg(parentBW.Encoding());
      p3.SetLastMotherDecayCoor(parentBW.Pos());
      p3.SetLastMotherDecayMom(parentBW.Mom());

      //set to daughters the same type as has mother  
      p1.SetType(NB);
      p2.SetType(NB);
      p3.SetType(NB);
      p1.SetPythiaStatusCode(parent.GetPythiaStatusCode());
      p2.SetPythiaStatusCode(parent.GetPythiaStatusCode());
      p3.SetPythiaStatusCode(parent.GetPythiaStatusCode());
            
      // energy conservation check in the lab system
      double deltaS = TMath::Sqrt((parentBW.Mom().X()-p1.Mom().X()-p2.Mom().X()-p3.Mom().X())*(parentBW.Mom().X()-p1.Mom().X()-p2.Mom().X()-p3.Mom().X()) +
                                  (parentBW.Mom().Y()-p1.Mom().Y()-p2.Mom().Y()-p3.Mom().Y())*(parentBW.Mom().Y()-p1.Mom().Y()-p2.Mom().Y()-p3.Mom().Y()) +
                                  (parentBW.Mom().Z()-p1.Mom().Z()-p2.Mom().Z()-p3.Mom().Z())*(parentBW.Mom().Z()-p1.Mom().Z()-p2.Mom().Z()-p3.Mom().Z())	+
                                  (parentBW.Mom().E()-p1.Mom().E()-p2.Mom().E()-p3.Mom().E())*(parentBW.Mom().E()-p1.Mom().E()-p2.Mom().E()-p3.Mom().E()));
      // if deltaS is too big then repeat the kinematic procedure
      if(deltaS>0.001) {
        
          LogDebug("HadronDecayer|deltaS") << "3-body decay kinematic check in lab system: " << pDef->GetPDG() << " >>> " << p1.Encoding() << " + " << p2.Encoding() << " + " << p3.Encoding() << endl
          << "Mother    (e,px,py,pz): " << parentBW.Mom().E() << " : " << parentBW.Mom().X() << " : " << parentBW.Mom().Y() << " : " << parentBW.Mom().Z() << endl
          << "Daughter1 (e,px,py,pz): " << p1.Mom().E() << " : " << p1.Mom().X() << " : " << p1.Mom().Y() << " : " << p1.Mom().Z() << endl
          << "Daughter2 (e,px,py,pz): " << p2.Mom().E() << " : " << p2.Mom().X() << " : " << p2.Mom().Y() << " : " << p2.Mom().Z() << endl
          << "Daughter3 (e,px,py,pz): " << p3.Mom().E() << " : " << p3.Mom().X() << " : " << p3.Mom().Y() << " : " << p3.Mom().Z() << endl
          << "3-body decay delta(sqrtS): " << deltaS << endl
          << "Repeating the decay algorithm";
        
	iterations++;
	continue;
      }

      // put particles in the list
      int parentIndex = parent.GetIndex();
      p1.SetIndex();
      p2.SetIndex();
      p3.SetIndex();
      p1.SetMother(parentIndex); 
      p2.SetMother(parentIndex);
      p3.SetMother(parentIndex);
      parent.SetFirstDaughterIndex(p1.GetIndex());
      parent.SetLastDaughterIndex(p3.GetIndex());
      allocator.AddParticle(p1, output);
      allocator.AddParticle(p2, output);
      allocator.AddParticle(p3, output);
      success = kTRUE;
    }
  }

  return;
}
