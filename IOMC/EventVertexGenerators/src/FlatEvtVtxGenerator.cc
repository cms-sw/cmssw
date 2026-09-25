// system includes
#include <numbers>
#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>
#include <CLHEP/Units/SystemOfUnits.h>

// user includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"

using CLHEP::cm;
using CLHEP::ns;
using CLHEP::radian;

FlatEvtVtxGenerator::FlatEvtVtxGenerator(const edm::ParameterSet& p)
    : BaseEvtVtxGenerator(p),
      fUseCylindricalCoords(p.getParameter<bool>("UseCylindricalCoords")),
      fMinZ(p.getParameter<double>("MinZ") * cm),
      fMaxZ(p.getParameter<double>("MaxZ") * cm),
      fMinT(p.getParameter<double>("MinT") * ns * c_light),
      fMaxT(p.getParameter<double>("MaxT") * ns * c_light) {
  if (fUseCylindricalCoords) {
    fMaxR = p.getParameter<double>("MaxR") * cm;
    fMinR = p.getParameter<double>("MinR") * cm;
    fMaxPhi = p.getParameter<double>("MaxPhi") * radian;
    fMinPhi = p.getParameter<double>("MinPhi") * radian;
    fMinX = std::nullopt;
    fMaxX = std::nullopt;
    fMinY = std::nullopt;
    fMaxY = std::nullopt;
  } else {
    fMinX = p.getParameter<double>("MinX") * cm;
    fMaxX = p.getParameter<double>("MaxX") * cm;
    fMinY = p.getParameter<double>("MinY") * cm;
    fMaxY = p.getParameter<double>("MaxY") * cm;
    fMaxR = std::nullopt;
    fMinR = std::nullopt;
    fMaxPhi = std::nullopt;
    fMinPhi = std::nullopt;
  }

  if (fMinZ > fMaxZ) {
    throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                          << "MinZ is greater than MaxZ";
  }
  if (fMinT > fMaxT) {
    throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                          << "MinT is greater than MaxT";
  }

  // configuration dependent checks
  if (fUseCylindricalCoords) {
    if (fMinR.value() > fMaxR.value()) {
      throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                            << "MinR is greater than MaxR";
    }
    if (fMinPhi.value() > fMaxPhi.value()) {
      throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                            << "MinPhi is greater than MaxPhi";
    }

    edm::LogVerbatim("FlatEvtVtx") << "FlatEvtVtxGenerator Initialized with r[" << *fMinR << ":" << *fMaxR
                                   << "] cm; phi[" << *fMinPhi << ":" << *fMaxPhi << "] rad; z[" << fMinZ << ":"
                                   << fMaxZ << "] cm; t[" << fMinT << ":" << fMaxT << "] mm";
  } else {
    if (fMinX.value() > fMaxX.value()) {
      throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                            << "MinX is greater than MaxX";
    }
    if (fMinY.value() > fMaxY.value()) {
      throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                            << "MinY is greater than MaxY";
    }

    edm::LogVerbatim("FlatEvtVtx") << "FlatEvtVtxGenerator Initialized with x[" << *fMinX << ":" << *fMaxX << "] cm; y["
                                   << *fMinY << ":" << *fMaxY << "] cm; z[" << fMinZ << ":" << fMaxZ << "] cm; t["
                                   << fMinT << ":" << fMaxT << "] mm" << std::endl;
  }
}

ROOT::Math::XYZTVector FlatEvtVtxGenerator::vertexShift(CLHEP::HepRandomEngine* engine) const {
  double aX, aY, aZ, aT;
  if (fUseCylindricalCoords) {
    double aR = CLHEP::RandFlat::shoot(engine, *fMinR, *fMaxR);
    double aPhi = CLHEP::RandFlat::shoot(engine, *fMinPhi, *fMaxPhi);
    aX = aR * std::cos(aPhi);
    aY = aR * std::sin(aPhi);
  } else {
    aX = CLHEP::RandFlat::shoot(engine, *fMinX, *fMaxX);
    aY = CLHEP::RandFlat::shoot(engine, *fMinY, *fMaxY);
  }
  aZ = CLHEP::RandFlat::shoot(engine, fMinZ, fMaxZ);
  aT = CLHEP::RandFlat::shoot(engine, fMinT, fMaxT);

  edm::LogVerbatim("FlatEvtVtx") << "FlatEvtVtxGenerator Vertex at [" << aX << ", " << aY << ", " << aZ << ", " << aT
                                 << "]";

  return ROOT::Math::XYZTVector(aX, aY, aZ, aT);
}

void FlatEvtVtxGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("MinZ", 0.0)->setComment("in cm");
  desc.add<double>("MaxZ", 0.001)->setComment("in cm");
  desc.add<double>("MinT", 0.0)->setComment("in ns");
  desc.add<double>("MaxT", 0.001)->setComment("in ns");
  desc.add<edm::InputTag>("src");

  desc.ifValue(edm::ParameterDescription<bool>("UseCylindricalCoords", false, true),
               // Cartesian (UseCylindricalCoords = false) branch
               (false >> (edm::ParameterDescription<double>("MinX", 0.0, true) and
                          edm::ParameterDescription<double>("MaxX", 0.001, true) and
                          edm::ParameterDescription<double>("MinY", 0.0, true) and
                          edm::ParameterDescription<double>("MaxY", 0.001, true))) or
                   // Cylindrical (UseCylindricalCoords = true) branch
                   (true >> (edm::ParameterDescription<double>("MinR", 0.0, true) and
                             edm::ParameterDescription<double>("MaxR", 0.001, true) and
                             edm::ParameterDescription<double>("MinPhi", -std::numbers::pi, true) and
                             edm::ParameterDescription<double>("MaxPhi", std::numbers::pi, true))));

  descriptions.addWithDefaultLabel(desc);
}
