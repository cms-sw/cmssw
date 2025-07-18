// $Id: HLLHCEvtVtxGenerator_Fix.cc, v 1.0 2015/03/15 10:32:11 Exp $

#include "IOMC/EventVertexGenerators/interface/HLLHCEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>

using namespace std;

namespace {
  constexpr double pmass = 0.9382720813e9;            // eV
  constexpr double gamma34 = 1.22541670246517764513;  // Gamma(3/4)
  constexpr double gamma14 = 3.62560990822190831193;  // Gamma(1/4)
  constexpr double gamma54 = 0.90640247705547798267;  // Gamma(5/4)
  constexpr double sqrt2 = 1.41421356237;
  constexpr double sqrt2to5 = 5.65685424949;
  constexpr double two_pi = 2.0 * M_PI;
}  // namespace

void HLLHCEvtVtxGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("MeanXIncm", 0.0);
  desc.add<double>("MeanYIncm", 0.0);
  desc.add<double>("MeanZIncm", 0.0);
  desc.add<double>("TimeOffsetInns", 0.0);
  desc.add<double>("EprotonInGeV", 7000.0);
  desc.add<double>("CrossingAngleInurad", 510.0);
  desc.add<double>("CrabbingAngleCrossingInurad", 380.0);
  desc.add<double>("CrabbingAngleSeparationInurad", 0.0);
  desc.add<double>("CrabFrequencyInMHz", 400.0);
  desc.add<bool>("RF800", false);
  desc.add<double>("BetaCrossingPlaneInm", 0.20);
  desc.add<double>("BetaSeparationPlaneInm", 0.20);
  desc.add<double>("HorizontalEmittance", 2.5e-06);
  desc.add<double>("VerticalEmittance", 2.05e-06);
  desc.add<double>("BunchLengthInm", 0.09);
  desc.add<edm::InputTag>("src");
  desc.add<bool>("readDB");
  descriptions.add("HLLHCEvtVtxGenerator", desc);
}

HLLHCEvtVtxGenerator::HLLHCEvtVtxGenerator(const edm::ParameterSet& p) : BaseEvtVtxGenerator(p) {
  readDB_ = p.getParameter<bool>("readDB");
  if (!readDB_) {
    // Read configurable parameters
    fMeanX = p.getParameter<double>("MeanXIncm") * CLHEP::cm;
    fMeanY = p.getParameter<double>("MeanYIncm") * CLHEP::cm;
    fMeanZ = p.getParameter<double>("MeanZIncm") * CLHEP::cm;
    fTimeOffset_c_light = p.getParameter<double>("TimeOffsetInns") * CLHEP::ns * CLHEP::c_light;
    fEProton = p.getParameter<double>("EprotonInGeV") * 1e9;
    fCrossingAngle = p.getParameter<double>("CrossingAngleInurad") * 1e-6;
    fCrabFrequency = p.getParameter<double>("CrabFrequencyInMHz") * 1e6;
    fRF800 = p.getParameter<bool>("RF800");
    fBetaCrossingPlane = p.getParameter<double>("BetaCrossingPlaneInm");
    fBetaSeparationPlane = p.getParameter<double>("BetaSeparationPlaneInm");
    fHorizontalEmittance = p.getParameter<double>("HorizontalEmittance");
    fVerticalEmittance = p.getParameter<double>("VerticalEmittance");
    fBunchLength = p.getParameter<double>("BunchLengthInm");
    fCrabbingAngleCrossing = p.getParameter<double>("CrabbingAngleCrossingInurad") * 1e-6;
    fCrabbingAngleSeparation = p.getParameter<double>("CrabbingAngleSeparationInurad") * 1e-6;
    // Set parameters inferred from configurables
    gamma = fEProton / pmass;
    beta = std::sqrt((1.0 - 1.0 / gamma) * ((1.0 + 1.0 / gamma)));
    betagamma = beta * gamma;
    oncc = fCrabbingAngleCrossing / fCrossingAngle;
    epsx = fHorizontalEmittance / (betagamma);
    epss = epsx;
    sigx = std::sqrt(epsx * fBetaCrossingPlane);
    phiCR = oncc * fCrossingAngle;
  }
  if (readDB_) {
    // NOTE: this is currently watching LS transitions, while it should watch Run transitions,
    // even though in reality there is no Run Dependent MC (yet) in CMS
    beamToken_ =
        esConsumes<SimBeamSpotHLLHCObjects, SimBeamSpotHLLHCObjectsRcd, edm::Transition::BeginLuminosityBlock>();
  }
}

void HLLHCEvtVtxGenerator::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& iEventSetup) {
  update(iEventSetup);
}

void HLLHCEvtVtxGenerator::update(const edm::EventSetup& iEventSetup) {
  if (readDB_ && parameterWatcher_.check(iEventSetup)) {
    edm::ESHandle<SimBeamSpotHLLHCObjects> beamhandle = iEventSetup.getHandle(beamToken_);
    // Read configurable parameters
    fMeanX = beamhandle->meanX() * CLHEP::cm;
    fMeanY = beamhandle->meanY() * CLHEP::cm;
    fMeanZ = beamhandle->meanZ() * CLHEP::cm;
    fEProton = beamhandle->eProton() * 1e9;
    fCrossingAngle = beamhandle->crossingAngle() * 1e-6;
    fCrabFrequency = beamhandle->crabFrequency() * 1e6;
    fRF800 = beamhandle->rf800();
    fBetaCrossingPlane = beamhandle->betaCrossingPlane();
    fBetaSeparationPlane = beamhandle->betaSeparationPlane();
    fHorizontalEmittance = beamhandle->horizontalEmittance();
    fVerticalEmittance = beamhandle->verticalEmittance();
    fBunchLength = beamhandle->bunchLenght();
    fCrabbingAngleCrossing = beamhandle->crabbingAngleCrossing() * 1e-6;
    fCrabbingAngleSeparation = beamhandle->crabbingAngleSeparation() * 1e-6;
    fTimeOffset_c_light = beamhandle->timeOffset() * CLHEP::ns * CLHEP::c_light;
    // Set parameters inferred from configurables
    gamma = fEProton / pmass;
    beta = std::sqrt((1.0 - 1.0 / gamma) * ((1.0 + 1.0 / gamma)));
    betagamma = beta * gamma;
    oncc = fCrabbingAngleCrossing / fCrossingAngle;
    epsx = fHorizontalEmittance / (betagamma);
    epss = epsx;
    sigx = std::sqrt(epsx * fBetaCrossingPlane);
    phiCR = oncc * fCrossingAngle;
  }
}

ROOT::Math::XYZTVector HLLHCEvtVtxGenerator::vertexShift(CLHEP::HepRandomEngine* engine) const {
  double imax = intensity(0., 0., 0., 0.);

  double x(0.), y(0.), z(0.), t(0.), i(0.);

  int count = 0;

  auto shoot = [&]() { return CLHEP::RandFlat::shoot(engine); };

  do {
    z = (shoot() - 0.5) * 6.0 * fBunchLength;
    t = (shoot() - 0.5) * 6.0 * fBunchLength;
    x = (shoot() - 0.5) * 12.0 * sigma(0.0, fHorizontalEmittance, fBetaCrossingPlane, betagamma);
    y = (shoot() - 0.5) * 12.0 * sigma(0.0, fVerticalEmittance, fBetaSeparationPlane, betagamma);

    i = intensity(x, y, z, t);

    if (i > imax)
      edm::LogError("Too large intensity") << "i>imax : " << i << " > " << imax << endl;
    ++count;
  } while ((i < imax * shoot()) && count < 10000);

  if (count > 9999)
    edm::LogError("Too many tries ") << " count : " << count << endl;

  //---convert to mm
  x *= 1000.0;
  y *= 1000.0;
  z *= 1000.0;
  t *= 1000.0;

  x += fMeanX;
  y += fMeanY;
  z += fMeanZ;
  t += fTimeOffset_c_light;

  return ROOT::Math::XYZTVector(x, y, z, t);
}

double HLLHCEvtVtxGenerator::sigma(double z, double epsilon, double beta, double betagamma) const {
  double sigma = std::sqrt(epsilon * (beta + z * z / beta) / betagamma);
  return sigma;
}

double HLLHCEvtVtxGenerator::intensity(double x, double y, double z, double t) const {
  //---c in m/s --- remember t is already in meters
  constexpr double c = 2.99792458e+8;  // m/s

  const double sigmay = sigma(z, fVerticalEmittance, fBetaSeparationPlane, betagamma);

  const double alphay_mod = fCrabbingAngleSeparation * std::cos(fCrabFrequency * (z - t) / c);

  const double cay = std::cos(alphay_mod);
  const double say = std::sin(alphay_mod);

  const double dy = -(z - t) * say - y * cay;

  const double xzt_density = integrandCC(x, z, t);

  const double norm = two_pi * sigmay;

  return (std::exp(-dy * dy / (sigmay * sigmay)) * xzt_density / norm);
}

double HLLHCEvtVtxGenerator::integrandCC(double x, double z, double ct) const {
  constexpr double local_c_light = 2.99792458e8;

  const double k = fCrabFrequency / local_c_light * two_pi;
  const double k2 = k * k;
  const double cos = std::cos(fCrossingAngle / 2.0);
  const double sin = std::sin(fCrossingAngle / 2.0);
  const double cos2 = cos * cos;
  const double sin2 = sin * sin;

  const double sigx2 = sigx * sigx;
  const double sigmax2 = sigx2 * (1 + z * z / (fBetaCrossingPlane * fBetaCrossingPlane));

  const double sigs2 = fBunchLength * fBunchLength;

  constexpr double factorRMSgauss4 =
      1. / sqrt2 / gamma34 * gamma14;  // # Factor to take rms sigma as input of the supergaussian
  constexpr double NormFactorGauss4 = sqrt2to5 * gamma54 * gamma54;

  const double sinCR = std::sin(phiCR / 2.0);
  const double sinCR2 = sinCR * sinCR;

  double result = -1.0;

  if (!fRF800) {
    const double norm = 2.0 / (two_pi * sigs2);
    const double cosks = std::cos(k * z);
    const double sinkct = std::sin(k * ct);
    result = norm *
             std::exp(-ct * ct / sigs2 - z * z * cos2 / sigs2 -
                      1.0 / (4 * k2 * sigmax2) *
                          (
                              //-4*cosks*cosks * sinkct*sinkct * sinCR2 // comes from integral over x
                              -8 * z * k * std::sin(k * z) * std::cos(k * ct) * sin * sinCR + 2 * sinCR2 -
                              std::cos(2 * k * (z - ct)) * sinCR2 - std::cos(2 * k * (z + ct)) * sinCR2 +
                              4 * k2 * z * z * sin2) -
                      x * x * (cos2 / sigmax2 + sin2 / sigs2)               // contribution from x integrand
                      + x * ct * sin / sigs2                                // contribution from x integrand
                      + 2 * x * cos * cosks * sinkct * sinCR / k / sigmax2  // contribution from x integrand
                      //+(2*ct/k)*np.cos(k*s)*np.sin(k*ct) *(sin*sinCR)/(sigs2*cos)  # small term
                      //+ct**2*(sin2/sigs4)/(cos2/sigmax2)                           # small term
                      ) /
             (1.0 + (z * z) / (fBetaCrossingPlane * fBetaCrossingPlane)) /
             std::sqrt(1.0 + (z * z) / (fBetaSeparationPlane * fBetaSeparationPlane));

  } else {
    const double norm = 2.0 / (NormFactorGauss4 * sigs2 * factorRMSgauss4);
    const double sigs4 = sigs2 * sigs2 * factorRMSgauss4 * factorRMSgauss4;
    const double cosks = std::cos(k * z);
    const double sinct = std::sin(k * ct);
    result =
        norm *
        std::exp(-ct * ct * ct * ct / sigs4 - z * z * z * z * cos2 * cos2 / sigs4 - 6 * ct * ct * z * z * cos2 / sigs4 -
                 sin2 / (4 * k2 * sigmax2) *
                     (2 + 4 * k2 * z * z - std::cos(2 * k * (z - ct)) - std::cos(2 * k * (z + ct)) -
                      8 * k * CLHEP::s * std::cos(k * ct) * std::sin(k * z) - 4 * cosks * cosks * sinct * sinct)) /
        std::sqrt((1 + z * z / (fBetaCrossingPlane * fBetaCrossingPlane)) /
                  (1 + z * z / (fBetaSeparationPlane * fBetaSeparationPlane)));
  }

  return result;
}
