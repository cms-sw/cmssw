
#include "IOMC/EventVertexGenerators/interface/GaussEvtVtxGenerator.h"
#include "Utilities/General/interface/CMSexception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>

using std::cout;
using std::endl;
using namespace edm;

GaussEvtVtxGenerator::GaussEvtVtxGenerator(const edm::ParameterSet & p )
: BaseEvtVtxGenerator(p)
{ 
  
  fRandom = new RandGauss(fEngine);
  
  fMeanX =  p.getParameter<double>("MeanX")*cm;
  fMeanY =  p.getParameter<double>("MeanY")*cm;
  fMeanZ =  p.getParameter<double>("MeanZ")*cm;
  fSigmaX = p.getParameter<double>("SigmaX")*cm;
  fSigmaY = p.getParameter<double>("SigmaY")*cm;
  fSigmaZ = p.getParameter<double>("SigmaZ")*cm;

  if (fSigmaX <= 0) {
    cout << "Error in GaussianEventVertexGenerator: "
	 << "Illegal resolution in X - set to default value 0.1cm (1mm)"
	 << endl;
    BaseGenexception ex("GaussianEventVertexGenerator:Illegal resolution in X");
    throw ex;
    fSigmaX = 0.1*cm; 
  }
  if (fSigmaY <= 0) {
    cout << "Error in GaussianEventVertexGenerator: "
	 << "Illegal resolution in Y - set to default value 0.1cm (1mm)"
	 << endl;
    BaseGenexception ex("GaussianEventVertexGenerator:Illegal resolution in Y");
    throw ex;
    fSigmaY = 0.1*cm; 
  }
  if (fSigmaZ <= 0) {
    cout << "Error in GaussianEventVertexGenerator: "
	 << "Illegal resolution in Z - set to default value 0.1cm (1mm)"
	 << endl;
    BaseGenexception ex("GaussianEventVertexGenerator:Illegal resolution in Z");
    throw ex;
    fSigmaZ = 0.1*cm; 
  }
    
}

GaussEvtVtxGenerator::~GaussEvtVtxGenerator() 
{
  // I'm not deleting this, since the engine seems to have
  // been delete earlier; thus an attempt tp delete RandGauss
  // results in a core dump... 
  // I need to ask Marc/Jim how to do it right...
  //delete myRandom; 
}

Hep3Vector* GaussEvtVtxGenerator::newVertex() {
  if ( fVertex != NULL ) delete fVertex;
  double X,Y,Z;
  X = fSigmaX * fRandom->fire() + fMeanX ;
  Y = fSigmaY * fRandom->fire() + fMeanY ;
  Z = fSigmaZ * fRandom->fire() + fMeanZ ;
  fVertex = new Hep3Vector(X, Y, Z);
  return fVertex;
}

void GaussEvtVtxGenerator::sigmaX(double s) 
{ 
  if (s>=0 ) {
    fSigmaX=s; 
  }
  else {
    cout << "Error in GaussianEventVertexGenerator: "
	 << "Illegal resolution in X - set to default value 0.1cm (1mm)"
	 << endl;
    BaseGenexception ex("GaussianEventVertexGenerator:Illegal resolution in X");
    throw ex;
    fSigmaX=0.1*cm;
  }
}

void GaussEvtVtxGenerator::sigmaY(double s) 
{ 
  if (s>=0 ) {
    fSigmaY=s; 
  }
  else {
    cout << "Error in GaussianEventVertexGenerator: "
	 << "Illegal resolution in Y - set to default value 0.1cm (1mm)"
	 << endl;
    BaseGenexception ex("GaussianEventVertexGenerator:Illegal resolution in Y");
    throw ex;
    fSigmaY=0.1*cm;
  }
}

void GaussEvtVtxGenerator::sigmaZ(double s) 
{ 
  if (s>=0 ) {
    fSigmaZ=s; 
  }
  else {
    cout << "Error in GaussianEventVertexGenerator: "
	 << "Illegal resolution in Z - set to default value 0.1cm (1mm)"
	 << endl;
    BaseGenexception ex("GaussianEventVertexGenerator:Illegal resolution in Z");
    throw ex;
    fSigmaZ=0.1*cm;
  }
}
