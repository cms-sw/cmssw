#include "IOMC/EventVertexGenerators/interface/GaussEvtVtxGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "HepMC/SimpleVector.h"

GaussEvtVtxGenerator::GaussEvtVtxGenerator(const edm::ParameterSet& p) : BaseEvtVtxGenerator(p) {
  readDB_ = p.getParameter<bool>("readDB");
  if (!readDB_) {
    fMeanX = p.getParameter<double>("MeanX") * cm;
    fMeanY = p.getParameter<double>("MeanY") * cm;
    fMeanZ = p.getParameter<double>("MeanZ") * cm;
    fSigmaX = p.getParameter<double>("SigmaX") * cm;
    fSigmaY = p.getParameter<double>("SigmaY") * cm;
    fSigmaZ = p.getParameter<double>("SigmaZ") * cm;
    fTimeOffset = p.getParameter<double>("TimeOffset") * ns * c_light;  // HepMC distance units are in mm

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
  if (readDB_) {
    // NOTE: this is currently watching LS transitions, while it should watch Run transitions,
    // even though in reality there is no Run Dependent MC (yet) in CMS
    beamToken_ = esConsumes<SimBeamSpotObjects, SimBeamSpotObjectsRcd, edm::Transition::BeginLuminosityBlock>();
  }
}

void GaussEvtVtxGenerator::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& iEventSetup) {
  update(iEventSetup);
}

void GaussEvtVtxGenerator::update(const edm::EventSetup& iEventSetup) {
  if (readDB_ && parameterWatcher_.check(iEventSetup)) {
    edm::ESHandle<SimBeamSpotObjects> beamhandle = iEventSetup.getHandle(beamToken_);
    fMeanX = beamhandle->meanX() * cm;
    fMeanY = beamhandle->meanY() * cm;
    fMeanZ = beamhandle->meanZ() * cm;
    fSigmaX = beamhandle->sigmaX() * cm;
    fSigmaY = beamhandle->sigmaY() * cm;
    fSigmaZ = beamhandle->sigmaZ() * cm;
    fTimeOffset = beamhandle->timeOffset() * ns * c_light;  // HepMC distance units are in mm
  }
}

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

void GaussEvtVtxGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("MeanX", 0.0)->setComment("in cm");
  desc.add<double>("MeanY", 0.0)->setComment("in cm");
  desc.add<double>("MeanZ", 0.0)->setComment("in cm");
  desc.add<double>("SigmaX", 0.0)->setComment("in cm");
  desc.add<double>("SigmaY", 0.0)->setComment("in cm");
  desc.add<double>("SigmaZ", 0.0)->setComment("in cm");
  desc.add<double>("TimeOffset", 0.0)->setComment("in ns");
  desc.add<edm::InputTag>("src");
  desc.add<bool>("readDB");
  descriptions.add("GaussEvtVtxGenerator", desc);
}
