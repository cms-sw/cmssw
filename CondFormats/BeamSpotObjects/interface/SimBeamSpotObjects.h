#ifndef SIMBEAMSPOTOBJECTS_H
#define SIMBEAMSPOTOBJECTS_H

/** \class SimBeamSpotObjects
 *
 * provide the vertex smearing parameters from DB
 *
 */

#include <sstream>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SimBeamSpotObjects {
 public:
  SimBeamSpotObjects(){};
  virtual ~ SimBeamSpotObjects(){};

  double fX0, fY0, fZ0;
  double fSigmaZ;
  double fbetastar, femittance;
  double fPhi,fAlpha;
  double fTimeOffset;

  void print(std::stringstream& ss) const;

  void read(edm::ParameterSet & p){
    fX0 =        p.getParameter<double>("X0")*cm;
    fY0 =        p.getParameter<double>("Y0")*cm;
    fZ0 =        p.getParameter<double>("Z0")*cm;
    fSigmaZ =    p.getParameter<double>("SigmaZ")*cm;
    fAlpha =     p.getParameter<double>("Alpha")*radian;
    fPhi =       p.getParameter<double>("Phi")*radian;
    fbetastar =  p.getParameter<double>("BetaStar")*cm;
    femittance = p.getParameter<double>("Emittance")*cm; // this is not the normalized emittance
    fTimeOffset = p.getParameter<double>("TimeOffset")*ns*c_light; // HepMC time units are mm
  }
};

std::ostream& operator<< ( std::ostream&, SimBeamSpotObjects beam );

#endif
