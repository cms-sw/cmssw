#ifndef Point5MaterialMap_cc
#define Point5MaterialMap_cc

#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"

inline bool inAir(double vx, double vy, double vz){
  bool air = false;
  // particles above surface of earth
  if (vy > SurfaceOfEarth) air = true;
  // CMS cavern (UXC 55)
  if (fabs(vz) < 26548.){
    if (sqrt((vx*1.1576)*(vx*1.1576) + vy*vy) < 15460.) air = true;
    if (vy < -8762.) air = false;
  }
  // access shaft (PX 56)
  if (vy > 0. && vy < (SurfaceOfEarth-2250.)){
    if (sqrt(vx*vx + (vz+Z_PX56)*(vz+Z_PX56)) < 10250.) air = true;
  }
  return air;
}

inline bool inWall(double vx, double vy, double vz){
  bool wall = false;
  // phase II surface building
  if (vy < SurfaceOfEarth && vy > (SurfaceOfEarth-2250.)){
    if (fabs(vz+Z_PX56) < 30000.){
      if (fabs(vx) < 10950) wall = true;
    }
    // foundation of crane
    if (fabs(vz+Z_PX56) < 9000.){
      if (fabs(vx) >= 10950 && fabs(vx) < 16950) wall = true;
    }
  }
  // CMS cavern (UXC 55)
  if (fabs(vz) < 29278.){
    if (sqrt((vx*1.1576)*(vx*1.1576) + vy*vy) < 16830.) wall = true;
    if (vy < -11762.) wall = false;
  }
  // access shaft (PX 56)
  if (vy > 0. && vy < (SurfaceOfEarth-2250.)){
    if (sqrt(vx*vx + (vz+Z_PX56)*(vz+Z_PX56)) < 12400.) wall = true;
  }
  if (inAir(vx,vy,vz)) wall = false;
  return wall;
}

inline bool inRock(double vx, double vy, double vz){
  bool rock = true;
  // check, if particle is not in an other material already
  if (inWall(vx,vy,vz)) rock = false;
  if (inAir(vx,vy,vz))  rock = false;
  return rock;
}

#endif
