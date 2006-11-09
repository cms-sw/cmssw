
// $Id: BetafuncEvtVtxGenerator.cc,v 1.0 2006/07/20 14:34:40 yumiceva Exp $
/*
________________________________________________________________________

 BetafuncEvtVtxGenerator

 Smear vertex according to the Beta function on the transverse plane
 and a Gaussian on the z axis. It allows the beam to have a crossing
 angle (slopes dxdz and dydz).

 Based on GaussEvtVtxGenerator
 implemented by Francisco Yumiceva (yumiceva@fnal.gov)

 FERMILAB
 2006
________________________________________________________________________
*/



#include "IOMC/EventVertexGenerators/interface/BetafuncEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Vector/ThreeVector.h"

#include <iostream>


BetafuncEvtVtxGenerator::BetafuncEvtVtxGenerator(const edm::ParameterSet & p )
: BaseEvtVtxGenerator(p)
{ 
  
  fRandom = new CLHEP::RandGauss(getEngine());

  fX0 =        p.getParameter<double>("X0")*cm;
  fY0 =        p.getParameter<double>("Y0")*cm;
  fZ0 =        p.getParameter<double>("Z0")*cm;
  fSigmaZ =    p.getParameter<double>("SigmaZ")*cm;
  fdxdz =      p.getParameter<double>("dxdz")*cm;
  fdydz =      p.getParameter<double>("dydz")*cm;
  fbetastar =  p.getParameter<double>("BetaStar")*cm;
  femmitance = p.getParameter<double>("Emmitance")*cm;
  
 
  if (fSigmaZ <= 0) {
	  throw cms::Exception("Configuration")
		  << "Error in BetafuncEvtVtxGenerator: "
		  << "Illegal resolution in Z (SigmaZ is negative)";
  }
    
}

BetafuncEvtVtxGenerator::~BetafuncEvtVtxGenerator() 
{
    delete fRandom; 
}

Hep3Vector* BetafuncEvtVtxGenerator::newVertex() {
	
	double X,Y,Z;
	
	double tmp_sigz = fRandom->fire(0., fSigmaZ);
	Z = tmp_sigz + fZ0;

	double tmp_sigx = BetaFunction(tmp_sigz,fZ0); 
	X = fRandom->fire(0.,tmp_sigx) + fX0 + Z*fdxdz ;

	double tmp_sigy = BetaFunction(tmp_sigz,fZ0);
	Y = fRandom->fire(0.,tmp_sigy) + fY0 + Z*fdydz;
	  
	if (fVertex == 0) fVertex = new CLHEP::Hep3Vector;
	
	fVertex->set(X, Y, Z);
		
	return fVertex;
}

double BetafuncEvtVtxGenerator::BetaFunction(double z, double z0)
{
	return sqrt(femmitance*(fbetastar+(((z-z0)*(z-z0))/fbetastar)));

}


void BetafuncEvtVtxGenerator::sigmaZ(double s) 
{ 
	if (s>=0 ) {
		fSigmaZ=s; 
	}
	else {
		throw cms::Exception("LogicError")
			<< "Error in BetafuncEvtVtxGenerator::sigmaZ: "
			<< "Illegal resolution in Z (negative)";
	}
}
