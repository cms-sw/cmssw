//-*-C++-*-
//-*-Particle.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////
#ifndef PARTICLE_HH
#define PARTICLE_HH

//#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Vector/LorentzVector.h"

namespace Exhume{
  class Particle{
  public:
    Particle(): p(0),vtx(0),id(0),Colour(0),AntiColour(0){
     
    }
    
    Particle(const CLHEP::HepLorentzVector& _p,const CLHEP::HepLorentzVector& _vtx, int _id,
	     int _Colour,int _AntiColour): p(_p),vtx(_vtx),id(_id),
					   Colour(_Colour),
					   AntiColour(_AntiColour){
     
    }
    
    ~Particle(){};
    //should make these private??????
    CLHEP::HepLorentzVector p;
    CLHEP::HepLorentzVector vtx;
    int id;
    int Colour;
    int AntiColour;
    
  };
}

#endif
/////////////////////////////////////////////////////////////////////////////
