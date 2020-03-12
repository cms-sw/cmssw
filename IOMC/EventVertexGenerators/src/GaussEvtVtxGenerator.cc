

#include "IOMC/EventVertexGenerators/interface/GaussEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "CLHEP/Vector/ThreeVector.h"
#include "HepMC/SimpleVector.h"

GaussEvtVtxGenerator::GaussEvtVtxGenerator(const edm::ParameterSet& p) : BaseEvtVtxGenerator(p) {
  fMeanX = p.getParameter<double>("MeanX") * cm;
  fMeanY = p.getParameter<double>("MeanY") * cm;
  fMeanZ = p.getParameter<double>("MeanZ") * cm;
  fSigmaX = p.getParameter<double>("SigmaX") * cm;
  fSigmaY = p.getParameter<double>("SigmaY") * cm;
  fSigmaZ = p.getParameter<double>("SigmaZ") * cm;
  fTimeOffset = p.getParameter<double>("TimeOffset") * ns * c_light;

  if (fSigmaX < 0) {
    throw cms::Exception("Configuration") << "Error in GaussEvtVtxGenerator: "
                                          << "Illegal resolution in X (SigmaX is negative)";
  }
  if (fSigmaY < 0) {
    throw cms::Exception("Configuration") << "Error in GaussEvtVtxGenerator: "
                                          << "Illegal resolution in Y (SigmaY is negative)";
  }
  if (fSigmaZ < 0) {
    throw cms::Exception("Configuration") << "Error in GaussEvtVtxGenerator: "
                                          << "Illegal resolution in Z (SigmaZ is negative)";
  }
}

GaussEvtVtxGenerator::~GaussEvtVtxGenerator() {}

HepMC::FourVector GaussEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine* engine) const {
  double X, Y, Z, T;
  X = CLHEP::RandGaussQ::shoot(engine, fMeanX, fSigmaX);
  Y = CLHEP::RandGaussQ::shoot(engine, fMeanY, fSigmaY);
  Z = CLHEP::RandGaussQ::shoot(engine, fMeanZ, fSigmaZ);
  T = CLHEP::RandGaussQ::shoot(engine, fTimeOffset, fSigmaZ);

  return HepMC::FourVector(X, Y, Z, T);
}

void GaussEvtVtxGenerator::sigmaX(double s) {
  if (s >= 0) {
    fSigmaX = s;
  } else {
    throw cms::Exception("LogicError") << "Error in GaussEvtVtxGenerator::sigmaX: "
                                       << "Illegal resolution in X (negative)";
  }
}

void GaussEvtVtxGenerator::sigmaY(double s) {
  if (s >= 0) {
    fSigmaY = s;
  } else {
    throw cms::Exception("LogicError") << "Error in GaussEvtVtxGenerator::sigmaY: "
                                       << "Illegal resolution in Y (negative)";
  }
}

void GaussEvtVtxGenerator::sigmaZ(double s) {
  if (s >= 0) {
    fSigmaZ = s;
  } else {
    throw cms::Exception("LogicError") << "Error in GaussEvtVtxGenerator::sigmaZ: "
                                       << "Illegal resolution in Z (negative)";
  }
}
