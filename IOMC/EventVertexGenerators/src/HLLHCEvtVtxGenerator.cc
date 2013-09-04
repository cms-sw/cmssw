// $Id: HLLHCEvtVtxGenerator.cc,v 1.4 2013/05/05 17:12:11 dlange Exp $

#include "IOMC/EventVertexGenerators/interface/HLLHCEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "HepMC/SimpleVector.h"

HLLHCEvtVtxGenerator::HLLHCEvtVtxGenerator(const edm::ParameterSet & p )
: BaseEvtVtxGenerator(p)
{ 
  
  fRandom = new CLHEP::RandGaussQ(getEngine());
  
  fMeanX =  p.getParameter<double>("MeanX")*cm;
  fMeanY =  p.getParameter<double>("MeanY")*cm;
  fMeanZ =  p.getParameter<double>("MeanZ")*cm;
  fSigmaX = p.getParameter<double>("SigmaX")*cm;
  fSigmaY = p.getParameter<double>("SigmaY")*cm;
  fSigmaZ = p.getParameter<double>("SigmaZ")*cm;
  fHalfCrossingAngle = p.getParameter<double>("HalfCrossingAngle");
  fCrabAngle = p.getParameter<double>("CrabAngle");
  fTimeOffset = p.getParameter<double>("TimeOffset")*ns*c_light;
  
  if (fSigmaX < 0) {
    throw cms::Exception("Configuration")
      << "Error in HLLHCEvtVtxGenerator: "
      << "Illegal resolution in X (SigmaX is negative)";
  }
  if (fSigmaY < 0) {
    throw cms::Exception("Configuration")
      << "Error in HLLHCEvtVtxGenerator: "
      << "Illegal resolution in Y (SigmaY is negative)";
  }
  if (fSigmaZ < 0) {
    throw cms::Exception("Configuration")
      << "Error in HLLHCEvtVtxGenerator: "
      << "Illegal resolution in Z (SigmaZ is negative)";
  }
}

HLLHCEvtVtxGenerator::~HLLHCEvtVtxGenerator() 
{
  delete fRandom; 
}


HepMC::FourVector* HLLHCEvtVtxGenerator::newVertex() {

  static const double sqrtOneHalf=sqrt(0.5);

  double sinbeta=sin(fCrabAngle);
  double cosbeta=cos(fCrabAngle);
  double cosalpha=cos(fHalfCrossingAngle);
  double sinamb=sin(fHalfCrossingAngle-fCrabAngle);
  double cosamb=cos(fHalfCrossingAngle-fCrabAngle);

  double SigmaX=sqrtOneHalf*hypot(fSigmaZ*sinbeta,fSigmaX*cosbeta)/cosalpha;

  double SigmaY=sqrtOneHalf*fSigmaY;

  double SigmaZ=sqrtOneHalf/hypot(sinamb/fSigmaX,cosamb/fSigmaZ);
  
  double X,Y,Z,T;
  X = SigmaX * fRandom->fire() + fMeanX ;
  Y = SigmaY * fRandom->fire() + fMeanY ;
  Z = SigmaZ * fRandom->fire() + fMeanZ ;

  double B=-2*X*c_light*(cosamb*sinbeta/(fSigmaX*fSigmaX)+
			 sinamb*cosbeta/(fSigmaZ*fSigmaZ));

  double C=c_light*c_light*(sinbeta*sinbeta/(fSigmaX*fSigmaX)+
			    cosbeta*cosbeta/(fSigmaZ*fSigmaZ));

  T = fTimeOffset-0.5*B/C+fRandom->fire()/sqrt(2*C);

  if ( fVertex == 0 ) fVertex = new HepMC::FourVector() ;

  //std::cout << "X Y Z T : "<<X<<" "<<Y<<" "<<Z<<" "<<T<<std::endl;

  fVertex->set( X, Y, Z, T );

  return fVertex;
}

void HLLHCEvtVtxGenerator::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("MeanX");
  desc.add<double>("MeanY");
  desc.add<double>("MeanZ");
  desc.add<double>("SigmaX");
  desc.add<double>("SigmaY");
  desc.add<double>("SigmaZ");
  desc.add<double>("HalfCrossingAngle");
  desc.add<double>("CrabAngle");
  desc.add<double>("TimeOffset");
  desc.add<edm::InputTag>("src");
  desc.add<bool>("readDB");
  descriptions.add("HLLHCEvtVtxGenerator",desc);

}

