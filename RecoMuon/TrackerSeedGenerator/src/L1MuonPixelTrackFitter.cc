#include "RecoMuon/TrackerSeedGenerator/interface/L1MuonPixelTrackFitter.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include <cmath>

template <class T>
T sqr(T t) {
  return t * t;
}

L1MuonPixelTrackFitter::L1MuonPixelTrackFitter(const edm::ParameterSet& cfg)
    : theConfig(cfg),
      invPtErrorScale{theConfig.getParameter<double>("invPtErrorScale")},
      phiErrorScale{theConfig.getParameter<double>("phiErrorScale")},
      cotThetaErrorScale{theConfig.getParameter<double>("cotThetaErrorScale")},
      tipErrorScale{theConfig.getParameter<double>("tipErrorScale")},
      zipErrorScale{theConfig.getParameter<double>("zipErrorScale")} {}

void L1MuonPixelTrackFitter::setL1Constraint(const L1MuGMTCand& muon) {
  thePhiL1 = muon.phiValue() + 0.021817;
  theEtaL1 = muon.etaValue();
  theChargeL1 = muon.charge();
}

void L1MuonPixelTrackFitter::setPxConstraint(const SeedingHitSet& hits) {
  theHit1 = hits[0]->globalPosition();
  theHit1 = hits[1]->globalPosition();
}

reco::Track* L1MuonPixelTrackFitter::run(const MagneticField& field,
                                         const std::vector<const TrackingRecHit*>& hits,
                                         const TrackingRegion& region) const {
  double phi_vtx_fromHits = (theHit2 - theHit1).phi();

  double invPt = valInversePt(phi_vtx_fromHits, thePhiL1, theEtaL1);
  double invPtErr = errInversePt(invPt, theEtaL1);

  int charge = (invPt > 0.) ? 1 : -1;
  double curvature = PixelRecoUtilities::curvature(invPt, field);

  Circle circle(theHit1, theHit2, curvature);

  double valPhi = this->valPhi(circle, charge);
  double valTip = this->valTip(circle, curvature);
  double valZip = this->valZip(curvature, theHit1, theHit2);
  double valCotTheta = this->valCotTheta(PixelRecoLineRZ(theHit1, theHit2));

  //  if ( (fabs(invPt)-0.1)/invPtErr > 3.) return 0;

  PixelTrackBuilder builder;
  double valPt = (fabs(invPt) > 1.e-4) ? 1. / fabs(invPt) : 1.e4;
  double errPt = invPtErrorScale * invPtErr * valPt * valPt;
  double eta = asinh(valCotTheta);
  double errPhi = this->errPhi(invPt, eta) * phiErrorScale;
  double errCotTheta = this->errCotTheta(invPt, eta) * cotThetaErrorScale;
  double errTip = this->errTip(invPt, eta) * tipErrorScale;
  double errZip = this->errZip(invPt, eta) * zipErrorScale;

  Measurement1D pt(valPt, errPt);
  Measurement1D phi(valPhi, errPhi);
  Measurement1D cotTheta(valCotTheta, errCotTheta);
  Measurement1D tip(valTip, errTip);
  Measurement1D zip(valZip, errZip);

  float chi2 = 0.;

  /*
  std::ostringstream str;
    str <<"\t ipt: " << invPt <<"+/-"<<invPtErr
        <<"\t pt:  " << pt.value() <<"+/-"<<pt.error()
        <<"\t phi: " << phi.value() <<"+/-"<<phi.error()
        <<"\t cot: " << cotTheta.value() <<"+/-"<<cotTheta.error()
        <<"\t tip: " << tip.value() <<"+/-"<<tip.error()
        <<"\t zip: " << zip.value() <<"+/-"<<zip.error()
        <<"\t charge: " << charge;
  std::cout <<str.str()<< std::endl;
*/

  return builder.build(pt, phi, cotTheta, tip, zip, chi2, charge, hits, &field);
}

double L1MuonPixelTrackFitter::valPhi(const Circle& circle, int charge) const {
  Circle::Point zero(0., 0., 0.);
  Circle::Vector center = circle.center() - zero;
  long double radius = center.perp();
  center /= radius;
  Circle::Vector dir = center.cross(-charge * Circle::Vector(0., 0., 1.));
  return dir.phi();
}

double L1MuonPixelTrackFitter::errPhi(double invPt, double eta) const {
  //
  // sigma = p0+p1/|pt|;
  //
  double p0, p1;
  int ieta = int(10 * fabs(eta));
  switch (ieta) {
    case 0: {
      p0 = 0.000597506;
      p1 = 0.00221057;
      break;
    }
    case 1: {
      p0 = 0.000591867;
      p1 = 0.00278744;
      break;
    }
    case 2: {
      p0 = 0.000635666;
      p1 = 0.00207433;
      break;
    }
    case 3: {
      p0 = 0.000619086;
      p1 = 0.00255121;
      break;
    }
    case 4: {
      p0 = 0.000572067;
      p1 = 0.00310618;
      break;
    }
    case 5: {
      p0 = 0.000596239;
      p1 = 0.00288442;
      break;
    }
    case 6: {
      p0 = 0.000607608;
      p1 = 0.00282996;
      break;
    }
    case 7: {
      p0 = 0.000606446;
      p1 = 0.00281118;
      break;
    }
    case 8: {
      p0 = 0.000594076;
      p1 = 0.00280546;
      break;
    }
    case 9: {
      p0 = 0.000579615;
      p1 = 0.00335534;
      break;
    }
    case 10: {
      p0 = 0.000659546;
      p1 = 0.00340443;
      break;
    }
    case 11: {
      p0 = 0.000659031;
      p1 = 0.00343151;
      break;
    }
    case 12: {
      p0 = 0.000738391;
      p1 = 0.00337297;
      break;
    }
    case 13: {
      p0 = 0.000798966;
      p1 = 0.00330008;
      break;
    }
    case 14: {
      p0 = 0.000702997;
      p1 = 0.00562643;
      break;
    }
    case 15: {
      p0 = 0.000973417;
      p1 = 0.00312666;
      break;
    }
    case 16: {
      p0 = 0.000995213;
      p1 = 0.00564278;
      break;
    }
    case 17: {
      p0 = 0.00121436;
      p1 = 0.00572704;
      break;
    }
    case 18: {
      p0 = 0.00119216;
      p1 = 0.00760204;
      break;
    }
    case 19: {
      p0 = 0.00141204;
      p1 = 0.0093777;
      break;
    }
    default: {
      p0 = 0.00153161;
      p1 = 0.00940265;
      break;
    }
  }
  return p0 + p1 * fabs(invPt);
}

double L1MuonPixelTrackFitter::valCotTheta(const PixelRecoLineRZ& line) const { return line.cotLine(); }

double L1MuonPixelTrackFitter::errCotTheta(double invPt, double eta) const {
  //
  // sigma = p0+p1/|pt|;
  //
  double p0, p1;
  int ieta = int(10 * fabs(eta));
  switch (ieta) {
    case 0: {
      p0 = 0.00166115;
      p1 = 5.75533e-05;
      break;
    }
    case 1: {
      p0 = 0.00157525;
      p1 = 0.000707437;
      break;
    }
    case 2: {
      p0 = 0.00122246;
      p1 = 0.000325456;
      break;
    }
    case 3: {
      p0 = 0.000852422;
      p1 = 0.000429216;
      break;
    }
    case 4: {
      p0 = 0.000637561;
      p1 = 0.00122298;
      break;
    }
    case 5: {
      p0 = 0.000555766;
      p1 = 0.00158096;
      break;
    }
    case 6: {
      p0 = 0.000641202;
      p1 = 0.00143339;
      break;
    }
    case 7: {
      p0 = 0.000803207;
      p1 = 0.000648816;
      break;
    }
    case 8: {
      p0 = 0.000741394;
      p1 = 0.0015289;
      break;
    }
    case 9: {
      p0 = 0.000652019;
      p1 = 0.00168873;
      break;
    }
    case 10: {
      p0 = 0.000716902;
      p1 = 0.00257556;
      break;
    }
    case 11: {
      p0 = 0.000800409;
      p1 = 0.00190563;
      break;
    }
    case 12: {
      p0 = 0.000808778;
      p1 = 0.00264139;
      break;
    }
    case 13: {
      p0 = 0.000775757;
      p1 = 0.00318478;
      break;
    }
    case 14: {
      p0 = 0.000705781;
      p1 = 0.00460576;
      break;
    }
    case 15: {
      p0 = 0.000580679;
      p1 = 0.00748248;
      break;
    }
    case 16: {
      p0 = 0.000561667;
      p1 = 0.00767487;
      break;
    }
    case 17: {
      p0 = 0.000521626;
      p1 = 0.0100178;
      break;
    }
    case 18: {
      p0 = 0.00064253;
      p1 = 0.0106062;
      break;
    }
    case 19: {
      p0 = 0.000636868;
      p1 = 0.0140047;
      break;
    }
    default: {
      p0 = 0.000682478;
      p1 = 0.0163569;
      break;
    }
  }
  return p0 + p1 * fabs(invPt);
}

double L1MuonPixelTrackFitter::valTip(const Circle& circle, double curvature) const {
  Circle::Point zero(0., 0., 0.);
  Circle::Vector center = circle.center() - zero;
  long double radius = center.perp();
  return radius - 1. / fabs(curvature);
}

double L1MuonPixelTrackFitter::errTip(double invPt, double eta) const {
  //
  // sigma = p0+p1/|pt|;
  //
  double p0, p1;
  int ieta = int(10 * fabs(eta));
  switch (ieta) {
    case 0: {
      p0 = 0.00392416;
      p1 = 0.00551809;
      break;
    }
    case 1: {
      p0 = 0.00390391;
      p1 = 0.00543244;
      break;
    }
    case 2: {
      p0 = 0.0040651;
      p1 = 0.00406496;
      break;
    }
    case 3: {
      p0 = 0.00387782;
      p1 = 0.00797637;
      break;
    }
    case 4: {
      p0 = 0.00376798;
      p1 = 0.00866894;
      break;
    }
    case 5: {
      p0 = 0.0042131;
      p1 = 0.00462184;
      break;
    }
    case 6: {
      p0 = 0.00392579;
      p1 = 0.00784685;
      break;
    }
    case 7: {
      p0 = 0.00370472;
      p1 = 0.00790174;
      break;
    }
    case 8: {
      p0 = 0.00364433;
      p1 = 0.00928368;
      break;
    }
    case 9: {
      p0 = 0.00387578;
      p1 = 0.00640431;
      break;
    }
    case 10: {
      p0 = 0.00382464;
      p1 = 0.00960763;
      break;
    }
    case 11: {
      p0 = 0.0038907;
      p1 = 0.0104562;
      break;
    }
    case 12: {
      p0 = 0.00392525;
      p1 = 0.0106442;
      break;
    }
    case 13: {
      p0 = 0.00400634;
      p1 = 0.011218;
      break;
    }
    case 14: {
      p0 = 0.0036229;
      p1 = 0.0156403;
      break;
    }
    case 15: {
      p0 = 0.00444317;
      p1 = 0.00832987;
      break;
    }
    case 16: {
      p0 = 0.00465492;
      p1 = 0.0179908;
      break;
    }
    case 17: {
      p0 = 0.0049652;
      p1 = 0.0216647;
      break;
    }
    case 18: {
      p0 = 0.0051395;
      p1 = 0.0233692;
      break;
    }
    case 19: {
      p0 = 0.0062917;
      p1 = 0.0262175;
      break;
    }
    default: {
      p0 = 0.00714444;
      p1 = 0.0253856;
      break;
    }
  }
  return p0 + p1 * fabs(invPt);
}

double L1MuonPixelTrackFitter::valZip(double curv, const GlobalPoint& pinner, const GlobalPoint& pouter) const {
  //
  // phi = asin(r*rho/2) with asin(x) ~= x+x**3/(2*3)
  //
  double rho3 = curv * curv * curv;
  double r1 = pinner.perp();
  double phi1 = r1 * curv / 2 + pinner.perp2() * r1 * rho3 / 48.;
  double r2 = pouter.perp();
  double phi2 = r2 * curv / 2 + pouter.perp2() * r2 * rho3 / 48.;
  double z1 = pinner.z();
  double z2 = pouter.z();

  return z1 - phi1 / (phi1 - phi2) * (z1 - z2);
}

double L1MuonPixelTrackFitter::errZip(double invPt, double eta) const {
  //
  // sigma = p0+p1/pt;
  //
  double p0, p1;
  int ieta = int(10 * fabs(eta));
  switch (ieta) {
    case 0: {
      p0 = 0.0120743;
      p1 = 0;
      break;
    }
    case 1: {
      p0 = 0.0110343;
      p1 = 0.0051199;
      break;
    }
    case 2: {
      p0 = 0.00846487;
      p1 = 0.00570084;
      break;
    }
    case 3: {
      p0 = 0.00668726;
      p1 = 0.00331165;
      break;
    }
    case 4: {
      p0 = 0.00467126;
      p1 = 0.00578239;
      break;
    }
    case 5: {
      p0 = 0.0043042;
      p1 = 0.00598517;
      break;
    }
    case 6: {
      p0 = 0.00515392;
      p1 = 0.00495422;
      break;
    }
    case 7: {
      p0 = 0.0060843;
      p1 = 0.00320512;
      break;
    }
    case 8: {
      p0 = 0.00564942;
      p1 = 0.00478876;
      break;
    }
    case 9: {
      p0 = 0.00532111;
      p1 = 0.0073239;
      break;
    }
    case 10: {
      p0 = 0.00579429;
      p1 = 0.00952782;
      break;
    }
    case 11: {
      p0 = 0.00614229;
      p1 = 0.00977795;
      break;
    }
    case 12: {
      p0 = 0.00714661;
      p1 = 0.00550482;
      break;
    }
    case 13: {
      p0 = 0.0066593;
      p1 = 0.00999362;
      break;
    }
    case 14: {
      p0 = 0.00634922;
      p1 = 0.0148156;
      break;
    }
    case 15: {
      p0 = 0.00600586;
      p1 = 0.0318022;
      break;
    }
    case 16: {
      p0 = 0.00676919;
      p1 = 0.027456;
      break;
    }
    case 17: {
      p0 = 0.00670066;
      p1 = 0.0317005;
      break;
    }
    case 18: {
      p0 = 0.00752392;
      p1 = 0.0347714;
      break;
    }
    case 19: {
      p0 = 0.00791425;
      p1 = 0.0566665;
      break;
    }
    default: {
      p0 = 0.00882372;
      p1 = 0.0596858;
      break;
    }
  }
  return p0 + p1 * fabs(invPt);
}

double L1MuonPixelTrackFitter::valInversePt(double phi0, double phiL1, double eta) const {
  double result = 0.;

  // solve equtaion p3*result^3 + p1*result + dphi = 0;
  // where:  result = 1/pt
  //         dphi = phiL1 - phi0
  // Cardan way is used

  double p1, p2, p3;  //parameters p1,p3 are negative by parameter fix in fit!
  param(eta, p1, p2, p3);

  double dphi = deltaPhi(phiL1, phi0);  // phi0-phiL1
  if (fabs(dphi) < 0.01) {
    result = -dphi / p1;
  } else {
    double q = dphi / 2. / p3;
    double p = p1 / 3. / p3;       // positive
    double D = q * q + p * p * p;  // positive
    double u = pow(-q + sqrt(D), 1. / 3.);
    double v = -pow(q + sqrt(D), 1. / 3.);
    result = u + v;
  }
  return result;
}

double L1MuonPixelTrackFitter::errInversePt(double invPt, double eta) const {
  //
  // pt*sigma(1/pt) = p0+p1*pt;
  //
  double p0, p1;
  int ieta = int(10 * fabs(eta));
  switch (ieta) {
    case 0: {
      p0 = 0.0196835;
      p1 = 0.00517533;
      break;
    }
    case 1: {
      p0 = 0.0266583;
      p1 = 0.00478101;
      break;
    }
    case 2: {
      p0 = 0.0217164;
      p1 = 0.00545425;
      break;
    }
    case 3: {
      p0 = 0.0197547;
      p1 = 0.00552263;
      break;
    }
    case 4: {
      p0 = 0.0208778;
      p1 = 0.00536009;
      break;
    }
    case 5: {
      p0 = 0.024192;
      p1 = 0.00521709;
      break;
    }
    case 6: {
      p0 = 0.0265315;
      p1 = 0.0051897;
      break;
    }
    case 7: {
      p0 = 0.0198071;
      p1 = 0.00566822;
      break;
    }
    case 8: {
      p0 = 0.0361955;
      p1 = 0.00486352;
      break;
    }
    case 9: {
      p0 = 0.037864;
      p1 = 0.00509094;
      break;
    }
    case 10: {
      p0 = 0.0382968;
      p1 = 0.00612354;
      break;
    }
    case 11: {
      p0 = 0.0308326;
      p1 = 0.0074234;
      break;
    }
    case 12: {
      p0 = 0.0248577;
      p1 = 0.00883049;
      break;
    }
    case 13: {
      p0 = 0.0279965;
      p1 = 0.00888293;
      break;
    }
    case 14: {
      p0 = 0.0372582;
      p1 = 0.00950252;
      break;
    }
    case 15: {
      p0 = 0.0281366;
      p1 = 0.0111501;
      break;
    }
    case 16: {
      p0 = 0.0421483;
      p1 = 0.0109413;
      break;
    }
    case 17: {
      p0 = 0.0461798;
      p1 = 0.0125824;
      break;
    }
    case 18: {
      p0 = 0.0530603;
      p1 = 0.0132638;
      break;
    }
    case 19: {
      p0 = 0.0601148;
      p1 = 0.0147911;
      break;
    }
    default: {
      p0 = 0.0552377;
      p1 = 0.0155574;
      break;
    }
  }
  return p1 + p0 * fabs(invPt);
}

double L1MuonPixelTrackFitter::findPt(double phi0, double phiL1, double eta, int charge) const {
  double dphi_min = fabs(deltaPhi(phi0, phiL1));
  double pt_best = 1.;
  double pt_cur = 1;
  while (pt_cur < 10000.) {
    double phi_exp = phi0 + getBending(1. / pt_cur, eta, charge);
    double dphi = fabs(deltaPhi(phi_exp, phiL1));
    if (dphi < dphi_min) {
      pt_best = pt_cur;
      dphi_min = dphi;
    }
    if (pt_cur < 10.)
      pt_cur += 0.01;
    else if (pt_cur < 20.)
      pt_cur += 0.025;
    else if (pt_cur < 100.)
      pt_cur += 0.1;
    else
      pt_cur += 1;
  };
  return pt_best;
}

double L1MuonPixelTrackFitter::getBending(double invPt, double eta, int charge) {
  double p1, p2, p3;
  param(eta, p1, p2, p3);
  return charge * p1 * invPt + charge * p2 * invPt * invPt + charge * p3 * invPt * invPt * invPt;
}

double L1MuonPixelTrackFitter::getBendingError(double invPt, double eta) {
  int ieta = int(10 * fabs(eta));
  double p0, p1;
  switch (ieta) {
    case 0: {
      p0 = 0.0196835;
      p1 = 0.00517533;
      break;
    }
    case 1: {
      p0 = 0.0266583;
      p1 = 0.00478101;
      break;
    }
    case 2: {
      p0 = 0.0217164;
      p1 = 0.00545425;
      break;
    }
    case 3: {
      p0 = 0.0197547;
      p1 = 0.00552263;
      break;
    }
    case 4: {
      p0 = 0.0208778;
      p1 = 0.00536009;
      break;
    }
    case 5: {
      p0 = 0.024192;
      p1 = 0.00521709;
      break;
    }
    case 6: {
      p0 = 0.0265315;
      p1 = 0.0051897;
      break;
    }
    case 7: {
      p0 = 0.0198071;
      p1 = 0.00566822;
      break;
    }
    case 8: {
      p0 = 0.0361955;
      p1 = 0.00486352;
      break;
    }
    case 9: {
      p0 = 0.037864;
      p1 = 0.00509094;
      break;
    }
    case 10: {
      p0 = 0.0382968;
      p1 = 0.00612354;
      break;
    }
    case 11: {
      p0 = 0.0308326;
      p1 = 0.0074234;
      break;
    }
    case 12: {
      p0 = 0.0248577;
      p1 = 0.00883049;
      break;
    }
    case 13: {
      p0 = 0.0279965;
      p1 = 0.00888293;
      break;
    }
    case 14: {
      p0 = 0.0372582;
      p1 = 0.00950252;
      break;
    }
    case 15: {
      p0 = 0.0281366;
      p1 = 0.0111501;
      break;
    }
    case 16: {
      p0 = 0.0421483;
      p1 = 0.0109413;
      break;
    }
    case 17: {
      p0 = 0.0461798;
      p1 = 0.0125824;
      break;
    }
    case 18: {
      p0 = 0.0530603;
      p1 = 0.0132638;
      break;
    }
    case 19: {
      p0 = 0.0601148;
      p1 = 0.0147911;
      break;
    }
    default: {
      p0 = 0.0552377;
      p1 = 0.0155574;
      break;
    }
  }
  return p0 + p1 * sqr(invPt);
}

void L1MuonPixelTrackFitter::param(double eta, double& p1, double& p2, double& p3) {
  int ieta = int(10 * fabs(eta));
  switch (ieta) {
    case 0: {
      p1 = -2.68016;
      p2 = 0;
      p3 = -12.9653;
      break;
    }
    case 1: {
      p1 = -2.67864;
      p2 = 0;
      p3 = -12.0036;
      break;
    }
    case 2: {
      p1 = -2.72997;
      p2 = 0;
      p3 = -10.3468;
      break;
    }
    case 3: {
      p1 = -2.68836;
      p2 = 0;
      p3 = -12.3369;
      break;
    }
    case 4: {
      p1 = -2.66885;
      p2 = 0;
      p3 = -11.589;
      break;
    }
    case 5: {
      p1 = -2.64932;
      p2 = 0;
      p3 = -12.7176;
      break;
    }
    case 6: {
      p1 = -2.64513;
      p2 = 0;
      p3 = -11.3912;
      break;
    }
    case 7: {
      p1 = -2.64487;
      p2 = 0;
      p3 = -9.83239;
      break;
    }
    case 8: {
      p1 = -2.58875;
      p2 = 0;
      p3 = -10.0541;
      break;
    }
    case 9: {
      p1 = -2.5551;
      p2 = 0;
      p3 = -9.75757;
      break;
    }
    case 10: {
      p1 = -2.55294;
      p2 = 0;
      p3 = -7.25238;
      break;
    }
    case 11: {
      p1 = -2.45333;
      p2 = 0;
      p3 = -7.966;
      break;
    }
    case 12: {
      p1 = -2.29283;
      p2 = 0;
      p3 = -7.03231;
      break;
    }
    case 13: {
      p1 = -2.13167;
      p2 = 0;
      p3 = -4.29182;
      break;
    }
    case 14: {
      p1 = -1.9102;
      p2 = 0;
      p3 = -3.21295;
      break;
    }
    case 15: {
      p1 = -1.74552;
      p2 = 0;
      p3 = -0.531827;
      break;
    }
    case 16: {
      p1 = -1.53642;
      p2 = 0;
      p3 = -1.74057;
      break;
    }
    case 17: {
      p1 = -1.39446;
      p2 = 0;
      p3 = -1.56819;
      break;
    }
    case 18: {
      p1 = -1.26176;
      p2 = 0;
      p3 = -0.598631;
      break;
    }
    case 19: {
      p1 = -1.14133;
      p2 = 0;
      p3 = -0.0941055;
      break;
    }
    default: {
      p1 = -1.02086;
      p2 = 0;
      p3 = -0.100491;
      break;
    }
  }
}

double L1MuonPixelTrackFitter::deltaPhi(double phi1, double phi2) const {
  while (phi1 >= 2 * M_PI)
    phi1 -= 2 * M_PI;
  while (phi2 >= 2 * M_PI)
    phi2 -= 2 * M_PI;
  while (phi1 < 0)
    phi1 += 2 * M_PI;
  while (phi2 < 0)
    phi2 += 2 * M_PI;
  double dPhi = phi2 - phi1;

  if (dPhi > M_PI)
    dPhi -= 2 * M_PI;
  if (dPhi < -M_PI)
    dPhi += 2 * M_PI;

  return dPhi;
}
