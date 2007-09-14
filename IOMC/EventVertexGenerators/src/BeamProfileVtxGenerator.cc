
// $Id: BeamProfileVtxGenerator.cc,v 1.3 2007/03/22 02:28:46 yarba Exp $

#include "IOMC/EventVertexGenerators/interface/BeamProfileVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/SystemOfUnits.h"
//#include "CLHEP/Vector/ThreeVector.h"
#include "HepMC/SimpleVector.h"


BeamProfileVtxGenerator::BeamProfileVtxGenerator(const edm::ParameterSet & p) :
  BaseEvtVtxGenerator(p), fRandom(0) {
  
  meanX(p.getUntrackedParameter<double>("BeamMeanX",0.0)*cm);
  meanY(p.getUntrackedParameter<double>("BeamMeanY",0.0)*cm);
  beamPos(p.getUntrackedParameter<double>("BeamPosition",0.0)*cm);
  sigmaX(p.getUntrackedParameter<double>("BeamSigmaX",0.0)*cm);
  sigmaY(p.getUntrackedParameter<double>("BeamSigmaY",0.0)*cm);
  double fMinEta = p.getUntrackedParameter<double>("MinEta",-5.5);
  double fMaxEta = p.getUntrackedParameter<double>("MaxEta",5.5);
  double fMinPhi = p.getUntrackedParameter<double>("MinPhi",-3.14159265358979323846);
  double fMaxPhi = p.getUntrackedParameter<double>("MaxPhi", 3.14159265358979323846);
  eta(0.5*(fMinEta+fMaxEta));
  phi(0.5*(fMinPhi+fMaxPhi));
  setType(p.getUntrackedParameter<bool>("GaussianProfile",true));

  edm::LogInfo("BeamProfileVtxGenerator") << "BeamProfileVtxGenerator: with beam "
				  << "along eta = " << fEta << " (Theta = " 
				  << fTheta/deg << ") phi = " << fPhi/deg 
				  << " centred at (" << fMeanX << ", " 
				  << fMeanY << ", "  << fMeanZ << ") and "
				  << "spread (" << fSigmaX << ", " << fSigmaY
				  << ") of type Gaussian = " << fType;
}

BeamProfileVtxGenerator::~BeamProfileVtxGenerator() {
  delete fRandom;
}

//Hep3Vector * BeamProfileVtxGenerator::newVertex() {
HepMC::FourVector* BeamProfileVtxGenerator::newVertex() {
  double aX, aY;
  if (fType) 
    aX = fSigmaX * (dynamic_cast<CLHEP::RandGaussQ*>(fRandom))->fire() + fMeanX;
  else
    aX = (dynamic_cast<CLHEP::RandFlat*>(fRandom))->fire(-0.5*fSigmaX,0.5*fSigmaX) + fMeanX ;
  if (fType) 
    aY = fSigmaY * (dynamic_cast<CLHEP::RandGaussQ*>(fRandom))->fire() + fMeanY;
  else
    aY = (dynamic_cast<CLHEP::RandFlat*>(fRandom))->fire(-0.5*fSigmaY,0.5*fSigmaY) + fMeanY;

  double xp = -aX*cos(fTheta)*cos(fPhi) +aY*sin(fPhi) +fMeanZ*sin(fTheta)*cos(fPhi);
  double yp = -aX*cos(fTheta)*sin(fPhi) -aY*cos(fPhi) +fMeanZ*sin(fTheta)*sin(fPhi);
  double zp =  aX*sin(fTheta)                          +fMeanZ*cos(fTheta);

  //if (fVertex == 0) fVertex = new CLHEP::Hep3Vector;
  //fVertex->set(xp, yp, zp);
  if (fVertex == 0 ) fVertex = new HepMC::FourVector() ;
  fVertex->set(xp, yp, zp, 0.);

//  LogDebug("BeamProfileVtxGenerator") << "BeamProfileVtxGenerator: Vertex created "
//			      << "at " << *fVertex;
  return fVertex;
}

void BeamProfileVtxGenerator::sigmaX(double s) { 

  if (s>=0) {
    fSigmaX = s; 
  } else {
    edm::LogWarning("BeamProfileVtxGenerator") << "Warning BeamProfileVtxGenerator:"
				       << " Illegal resolution in X " << s
				       << "- set to default value 0 cm";
    fSigmaX = 0;
  }
}

void BeamProfileVtxGenerator::sigmaY(double s) { 

  if (s>=0) {
    fSigmaY = s; 
  } else {
    edm::LogWarning("BeamProfileVtxGenerator") << "Warning BeamProfileVtxGenerator:"
				       << " Illegal resolution in Y " << s
				       << "- set to default value 0 cm";
    fSigmaY = 0;
  }
}

void BeamProfileVtxGenerator::eta(double s) { 
  fEta   = s; 
  fTheta = 2.0*atan(exp(-fEta));
}

void BeamProfileVtxGenerator::setType(bool s) { 

  fType = s;
  delete fRandom;
  
  if (fType == true)
    fRandom = new CLHEP::RandGaussQ(getEngine());
  else
    fRandom = new CLHEP::RandFlat(getEngine());
}
