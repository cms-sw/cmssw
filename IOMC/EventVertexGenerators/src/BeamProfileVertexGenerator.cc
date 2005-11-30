#include "IOMC/EventVertexGenerators/interface/BeamProfileVertexGenerator.h"
#include "Utilities/General/interface/CMSexception.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>

using std::cout;
using std::endl;

BeamProfileVertexGenerator::BeamProfileVertexGenerator(const edm::ParameterSet & p) 
: BaseEventVertexGenerator(p), m_pBeamProfileVertexGenerator(p), myVertex(0)
{  
  myMeanX = m_pBeamProfileVertexGenerator.getParameter<double>("BeamMeanX")*mm;
  myMeanY = m_pBeamProfileVertexGenerator.getParameter<double>("BeamMeanY")*mm;
  myMeanZ = m_pBeamProfileVertexGenerator.getParameter<double>("BeamPosition")*mm;
  mySigmaX = m_pBeamProfileVertexGenerator.getParameter<double>("BeamSigmaX")*mm;
  mySigmaY = m_pBeamProfileVertexGenerator.getParameter<double>("BeamSigmaY")*mm;
  myEta = m_pBeamProfileVertexGenerator.getParameter<double>("BeamEta");
  myPhi = m_pBeamProfileVertexGenerator.getParameter<double>("BeamPhi");
  myType = m_pBeamProfileVertexGenerator.getParameter<bool>("GaussianProfile");

  cout << "BeamProfileVertexGenerator: with beam along eta = " 
       << myEta << " (Theta = " << myTheta/deg << ") phi = " 
       << myPhi/deg << " centred at (" << myMeanX << ", " << myMeanY 
       << ", "  << myMeanZ << ") and spread (" << mySigmaX << ", "
       << mySigmaY << ") of type Gaussian = " << myType << endl;
}

BeamProfileVertexGenerator::~BeamProfileVertexGenerator() {
  if (myVertex) delete myVertex;
}

Hep3Vector * BeamProfileVertexGenerator::newVertex() {
  if (myVertex) delete myVertex;
  double aX, aY;
  if (myType) 
    aX = mySigmaX * RandGauss::shoot() + myMeanX;
  else
    aX = RandFlat::shoot(-0.5*mySigmaX,0.5*mySigmaX) + myMeanX;
  double tX = 90.*deg + myTheta;
  double sX = sin(tX);
  if (abs(sX)>1.e-12) sX = 1./sX;
  else                sX = 1.;
  double fX = atan2(sX*cos(myTheta)*sin(myPhi),sX*cos(myTheta)*cos(myPhi));
  if (myType) 
    aY = mySigmaY * RandGauss::shoot() + myMeanY;
  else
    aY = RandFlat::shoot(-0.5*mySigmaY,0.5*mySigmaY) + myMeanY;
  double fY = 90.*deg + myPhi;
  double xp = aX*sin(tX)*cos(fX) +aY*cos(fY) +myMeanZ*sin(myTheta)*cos(myPhi);
  double yp = aX*sin(tX)*sin(fX) +aY*cos(fY) +myMeanZ*sin(myTheta)*sin(myPhi);
  double zp = aX*cos(tX)                     +myMeanZ*cos(myTheta);

  myVertex = new Hep3Vector(xp, yp, zp);
  cout << "BeamProfileVertexGenerator: Vertex created at " << *myVertex
       << endl;
  return myVertex;
}

Hep3Vector * BeamProfileVertexGenerator::lastVertex() {
  return myVertex;
}

void BeamProfileVertexGenerator::sigmaX(double s) { 
  if (s>=0) {
    mySigmaX = s; 
  } else {
    cout << "Error in BeamProfileVertexGenerator: "
	 << "Illegal resolution in X - set to default value 0 mm"
	 << endl;
    BaseGenexception ex("BeamProfileVertexGenerator:Illegal resolution in X");
    throw ex;
    mySigmaX = 0;
  }
}

void BeamProfileVertexGenerator::sigmaY(double s) { 
  if (s>=0) {
    mySigmaY = s; 
  } else {
    cout << "Error in BeamProfileVertexGenerator: "
	 << "Illegal resolution in Y - set to default value 0 mm"
	 << endl;
    BaseGenexception ex("BeamProfileVertexGenerator:Illegal resolution in Y");
    throw ex;
    mySigmaY = 0;
  }
}

void BeamProfileVertexGenerator::eta(double s) { 
  myEta   = s; 
  myTheta = 2.0*atan(exp(-myEta));
}
