
// $Id: BeamProfileVtxGenerator.cc,v 1.5 2007/11/02 21:40:34 sunanda Exp $

#include "IOMC/EventVertexGenerators/interface/BeamProfileVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"
//#include "CLHEP/Vector/ThreeVector.h"
#include "HepMC/SimpleVector.h"

#include<fstream>
#include<string>

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
  nBinx = p.getUntrackedParameter<int>("BinX",50);
  nBiny = p.getUntrackedParameter<int>("BinY",50);
  ffile = p.getUntrackedParameter<bool>("UseFile",false);
  fTimeOffset = p.getParameter<double>("TimeOffset")*ns*c_light;
  
  if (ffile) {
    std::string file = p.getUntrackedParameter<std::string>("File","beam.profile");
    ifstream is(file.c_str(), std::ios::in);
    if (is) {
      double elem,sum=0;
      while (!is.eof()) {
	is >> elem;
        fdistn.push_back(elem);
	sum += elem;
      }
      if (((int)(fdistn.size())) >= nBinx*nBiny) {
	double last = 0;
	for (unsigned int i=0; i<fdistn.size(); i++) {
	  fdistn[i] /= sum;
	  fdistn[i] += last;
	  last       = fdistn[i];
	}
	setType(false);
      } else {
	ffile = false;
      }
    } else {
      ffile = false;
    }
  } 
  if (!ffile) {
    setType(p.getUntrackedParameter<bool>("GaussianProfile",true));
  }

  edm::LogInfo("BeamProfileVtxGenerator") << "BeamProfileVtxGenerator: with "
					  << "beam along eta = " << fEta 
					  << " (Theta = " << fTheta/deg 
					  << ") phi = " << fPhi/deg 
					  << " centred at (" << fMeanX << ", " 
					  << fMeanY << ", "  << fMeanZ << ") "
					  << "and spread (" << fSigmaX << ", "
					  << fSigmaY << ") of type Gaussian = "
					  << fType << " use file " << ffile;
  if (ffile) 
    edm::LogInfo("BeamProfileVtxGenerator") << "With " << nBinx << " bins "
					    << " along X and " << nBiny 
					    << " bins along Y";
}

BeamProfileVtxGenerator::~BeamProfileVtxGenerator() {
  delete fRandom;
}


//Hep3Vector * BeamProfileVtxGenerator::newVertex() {
HepMC::FourVector* BeamProfileVtxGenerator::newVertex() {
  double aX, aY;
  if (ffile) {
    double r1 = (dynamic_cast<CLHEP::RandFlat*>(fRandom))->fire();
    int ixy = 0, ix, iy;
    for (unsigned int i=0; i<fdistn.size(); i++) {
      if (r1 > fdistn[i]) ixy = i+1;
    }
    if (ixy >= nBinx*nBiny) {
      ix = nBinx-1; iy = nBiny-1;
    } else {
      ix = ixy%nBinx; iy = (ixy-ix)/nBinx;
    }
    aX = 0.5*(2*ix-nBinx+2*(dynamic_cast<CLHEP::RandFlat*>(fRandom))->fire())*fSigmaX + fMeanX ;
    aY = 0.5*(2*iy-nBiny+2*(dynamic_cast<CLHEP::RandFlat*>(fRandom))->fire())*fSigmaY + fMeanY ;
  } else {
    if (fType) {
      aX = fSigmaX*(dynamic_cast<CLHEP::RandGaussQ*>(fRandom))->fire() +fMeanX;
      aY = fSigmaY*(dynamic_cast<CLHEP::RandGaussQ*>(fRandom))->fire() +fMeanY;
    } else {
      aX = (dynamic_cast<CLHEP::RandFlat*>(fRandom))->fire(-0.5*fSigmaX,0.5*fSigmaX) + fMeanX ;
      aY = (dynamic_cast<CLHEP::RandFlat*>(fRandom))->fire(-0.5*fSigmaY,0.5*fSigmaY) + fMeanY;
    }
  }
  double xp = -aX*cos(fTheta)*cos(fPhi) +aY*sin(fPhi) +fMeanZ*sin(fTheta)*cos(fPhi);
  double yp = -aX*cos(fTheta)*sin(fPhi) -aY*cos(fPhi) +fMeanZ*sin(fTheta)*sin(fPhi);
  double zp =  aX*sin(fTheta)                         +fMeanZ*cos(fTheta);

  //if (fVertex == 0) fVertex = new CLHEP::Hep3Vector;
  //fVertex->set(xp, yp, zp);
  if (fVertex == 0 ) fVertex = new HepMC::FourVector() ;
  fVertex->set(xp, yp, zp, fTimeOffset );

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
