//FAMOS headers
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include <array>

BaseParticlePropagator::BaseParticlePropagator() : rCyl(0.), rCyl2(0.), zCyl(0.), bField(0), properDecayTime(1E99) { ; }

BaseParticlePropagator::BaseParticlePropagator(const RawParticle& myPart, double R, double Z, double B, double t)
    : particle_(myPart), rCyl(R), rCyl2(R * R), zCyl(Z), bField(B), properDecayTime(t) {
  init();
}

BaseParticlePropagator::BaseParticlePropagator(const RawParticle& myPart, double R, double Z, double B)
    : particle_(myPart), rCyl(R), rCyl2(R * R), zCyl(Z), bField(B), properDecayTime(1E99) {
  init();
}

void BaseParticlePropagator::init() {
  // The status of the propagation
  success = 0;
  // Propagate only for the first half-loop
  firstLoop = true;
  //
  fiducial = true;
  // The particle has not yet decayed
  decayed = false;
  // The proper time is set to zero
  properTime = 0.;
  // The propagation direction is along the momentum
  propDir = 1;
  //
  debug = false;
}

bool BaseParticlePropagator::propagate() {
  //
  // Check that the particle is not already on the cylinder surface
  //
  double rPos2 = particle_.R2();

  if (onBarrel(rPos2)) {
    success = 1;
    return true;
  }
  //
  if (onEndcap(rPos2)) {
    success = 2;
    return true;
  }
  //
  // Treat neutral particles (or no magnetic field) first
  //
  if (fabs(particle_.charge()) < 1E-12 || bField < 1E-12) {
    //
    // First check that the particle crosses the cylinder
    //
    double pT2 = particle_.Perp2();
    //    double pT2 = pT*pT;
    //    double b2 = rCyl * pT;
    double b2 = rCyl2 * pT2;
    double ac = fabs(particle_.X() * particle_.Py() - particle_.Y() * particle_.Px());
    double ac2 = ac * ac;
    //
    // The particle never crosses (or never crossed) the cylinder
    //
    //    if ( ac > b2 ) return false;
    if (ac2 > b2)
      return false;

    //    double delta  = std::sqrt(b2*b2 - ac*ac);
    double delta = std::sqrt(b2 - ac2);
    double tplus = -(particle_.X() * particle_.Px() + particle_.Y() * particle_.Py()) + delta;
    double tminus = -(particle_.X() * particle_.Px() + particle_.Y() * particle_.Py()) - delta;
    //
    // Find the first (time-wise) intersection point with the cylinder
    //

    double solution = tminus < 0 ? tplus / pT2 : tminus / pT2;
    if (solution < 0.)
      return false;
    //
    // Check that the particle does (did) not exit from the endcaps first.
    //
    double zProp = particle_.Z() + particle_.Pz() * solution;
    if (fabs(zProp) > zCyl) {
      tplus = (zCyl - particle_.Z()) / particle_.Pz();
      tminus = (-zCyl - particle_.Z()) / particle_.Pz();
      solution = tminus < 0 ? tplus : tminus;
      if (solution < 0.)
        return false;
      success = 2;
    } else {
      success = 1;
    }

    //
    // Check the decay time
    //
    double delTime = propDir * particle_.mass() * solution;
    double factor = 1.;
    properTime += delTime;
    if (properTime > properDecayTime) {
      factor = 1. - (properTime - properDecayTime) / delTime;
      properTime = properDecayTime;
      decayed = true;
    }

    //
    // Compute the coordinates of the RawParticle after propagation
    //
    double xProp = particle_.X() + particle_.Px() * solution * factor;
    double yProp = particle_.Y() + particle_.Py() * solution * factor;
    zProp = particle_.Z() + particle_.Pz() * solution * factor;
    double tProp = particle_.T() + particle_.E() * solution * factor;

    //
    // Last check : Although propagated to the endcaps, R could still be
    // larger than rCyl because the cylinder was too short...
    //
    if (success == 2 && xProp * xProp + yProp * yProp > rCyl2) {
      success = -1;
      return true;
    }

    //
    // ... and update the particle with its propagated coordinates
    //
    particle_.setVertex(XYZTLorentzVector(xProp, yProp, zProp, tProp));

    return true;
  }
  //
  // Treat charged particles with magnetic field next.
  //
  else {
    //
    // First check that the particle can cross the cylinder
    //
    double pT = particle_.Pt();
    double radius = helixRadius(pT);
    double phi0 = helixStartPhi();
    double xC = helixCentreX(radius, phi0);
    double yC = helixCentreY(radius, phi0);
    double dist = helixCentreDistToAxis(xC, yC);
    //
    // The helix either is outside or includes the cylinder -> no intersection
    //
    if (fabs(fabs(radius) - dist) > rCyl)
      return false;
    //
    // The particle is already away from the endcaps, and will never come back
    // Could be back-propagated, so warn the user that it is so.
    //
    if (particle_.Z() * particle_.Pz() > zCyl * fabs(particle_.Pz())) {
      success = -2;
      return true;
    }

    double pZ = particle_.Pz();
    double phiProp, zProp;
    //
    // Does the helix cross the cylinder barrel ? If not, do the first half-loop
    //
    double rProp = std::min(fabs(radius) + dist - 0.000001, rCyl);

    // Check for rounding errors in the ArcSin.
    double sinPhiProp = (rProp * rProp - radius * radius - dist * dist) / (2. * dist * radius);

    //
    double deltaPhi = 1E99;

    // Taylor development up to third order for large momenta
    if (1. - fabs(sinPhiProp) < 1E-9) {
      double cphi0 = std::cos(phi0);
      double sphi0 = std::sin(phi0);
      double r0 = (particle_.X() * cphi0 + particle_.Y() * sphi0) / radius;
      double q0 = (particle_.X() * sphi0 - particle_.Y() * cphi0) / radius;
      double rcyl2 = (rCyl2 - particle_.X() * particle_.X() - particle_.Y() * particle_.Y()) / (radius * radius);
      double delta = r0 * r0 + rcyl2 * (1. - q0);

      // This is a solution of a second order equation, assuming phi = phi0 + epsilon
      // and epsilon is small.
      deltaPhi = radius > 0 ? (-r0 + std::sqrt(delta)) / (1. - q0) : (-r0 - std::sqrt(delta)) / (1. - q0);
    }

    // Use complete calculation otherwise, or if the delta phi is large anyway.
    if (fabs(deltaPhi) > 1E-3) {
      //    phiProp =  fabs(sinPhiProp) < 0.99999999  ?
      //      std::asin(sinPhiProp) : M_PI/2. * sinPhiProp;
      phiProp = std::asin(sinPhiProp);

      //
      // Compute phi after propagation (two solutions, but the asin definition
      // (between -pi/2 and +pi/2) takes care of the wrong one.
      //
      phiProp = helixCentrePhi(xC, yC) + (inside(rPos2) || onSurface(rPos2) ? phiProp : M_PI - phiProp);

      //
      // Solve the obvious two-pi ambiguities: more than one turn!
      //
      if (fabs(phiProp - phi0) > 2. * M_PI)
        phiProp += (phiProp > phi0 ? -2. * M_PI : +2. * M_PI);

      //
      // Check that the phi difference has the right sign, according to Q*B
      // (Another two pi ambuiguity, slightly more subtle)
      //
      if ((phiProp - phi0) * radius < 0.0)
        radius > 0.0 ? phiProp += 2 * M_PI : phiProp -= 2 * M_PI;

    } else {
      // Use Taylor
      phiProp = phi0 + deltaPhi;
    }

    //
    // Check that the particle does not exit from the endcaps first.
    //
    zProp = particle_.Z() + (phiProp - phi0) * pZ * radius / pT;
    if (fabs(zProp) > zCyl) {
      zProp = zCyl * fabs(pZ) / pZ;
      phiProp = phi0 + (zProp - particle_.Z()) / radius * pT / pZ;
      success = 2;

    } else {
      //
      // If the particle does not cross the barrel, either process the
      // first-half loop only or propagate all the way down to the endcaps
      //
      if (rProp < rCyl) {
        if (firstLoop || fabs(pZ) / pT < 1E-10) {
          success = 0;

        } else {
          zProp = zCyl * fabs(pZ) / pZ;
          phiProp = phi0 + (zProp - particle_.Z()) / radius * pT / pZ;
          success = 2;
        }
        //
        // The particle crossed the barrel
        //
      } else {
        success = 1;
      }
    }
    //
    // Compute the coordinates of the RawParticle after propagation
    //
    //
    // Check the decay time
    //
    double delTime = propDir * (phiProp - phi0) * radius * particle_.mass() / pT;
    double factor = 1.;
    properTime += delTime;
    if (properTime > properDecayTime) {
      factor = 1. - (properTime - properDecayTime) / delTime;
      properTime = properDecayTime;
      decayed = true;
    }

    zProp = particle_.Z() + (zProp - particle_.Z()) * factor;

    phiProp = phi0 + (phiProp - phi0) * factor;

    double sProp = std::sin(phiProp);
    double cProp = std::cos(phiProp);
    double xProp = xC + radius * sProp;
    double yProp = yC - radius * cProp;
    double tProp = particle_.T() + (phiProp - phi0) * radius * particle_.E() / pT;
    double pxProp = pT * cProp;
    double pyProp = pT * sProp;

    //
    // Last check : Although propagated to the endcaps, R could still be
    // larger than rCyl because the cylinder was too short...
    //
    if (success == 2 && xProp * xProp + yProp * yProp > rCyl2) {
      success = -1;
      return true;
    }
    //
    // ... and update the particle vertex and momentum
    //
    particle_.setVertex(XYZTLorentzVector(xProp, yProp, zProp, tProp));
    particle_.setMomentum(pxProp, pyProp, pZ, particle_.E());
    //    particle_.SetPx(pxProp);
    //    particle_.SetPy(pyProp);
    return true;
  }
}

bool BaseParticlePropagator::backPropagate() {
  // Backpropagate
  particle_.setMomentum(-particle_.Px(), -particle_.Py(), -particle_.Pz(), particle_.E());
  particle_.setCharge(-particle_.charge());
  propDir = -1;
  bool done = propagate();
  particle_.setMomentum(-particle_.Px(), -particle_.Py(), -particle_.Pz(), particle_.E());
  particle_.setCharge(-particle_.charge());
  propDir = +1;

  return done;
}

BaseParticlePropagator BaseParticlePropagator::propagated() const {
  // Copy the input particle in a new instance
  BaseParticlePropagator myPropPart(*this);
  // Allow for many loops
  myPropPart.firstLoop = false;
  // Propagate the particle !
  myPropPart.propagate();
  // Back propagate if the forward propagation was not a success
  if (myPropPart.success < 0)
    myPropPart.backPropagate();
  // Return the new instance
  return myPropPart;
}

void BaseParticlePropagator::setPropagationConditions(double R, double Z, bool f) {
  rCyl = R;
  rCyl2 = R * R;
  zCyl = Z;
  firstLoop = f;
}

bool BaseParticlePropagator::propagateToClosestApproach(double x0, double y0, bool first) {
  // Pre-computed quantities
  double pT = particle_.Pt();
  double radius = helixRadius(pT);
  double phi0 = helixStartPhi();

  // The Helix centre coordinates are computed wrt to the z axis
  // Actually the z axis might be centred on 0,0: it's the beam spot position (x0,y0)!
  double xC = helixCentreX(radius, phi0);
  double yC = helixCentreY(radius, phi0);
  double distz = helixCentreDistToAxis(xC - x0, yC - y0);
  double dist0 = helixCentreDistToAxis(xC, yC);

  //
  // Propagation to point of clostest approach to z axis
  //
  double rz, r0, z;
  if (particle_.charge() != 0.0 && bField != 0.0) {
    rz = fabs(fabs(radius) - distz) + std::sqrt(x0 * x0 + y0 * y0) + 0.0000001;
    r0 = fabs(fabs(radius) - dist0) + 0.0000001;
  } else {
    rz = fabs(particle_.Px() * (particle_.Y() - y0) - particle_.Py() * (particle_.X() - x0)) / particle_.Pt();
    r0 = fabs(particle_.Px() * particle_.Y() - particle_.Py() * particle_.X()) / particle_.Pt();
  }

  z = 999999.;

  // Propagate to the first interesection point
  // with cylinder of radius sqrt(x0**2+y0**2)
  setPropagationConditions(rz, z, first);
  bool done = backPropagate();

  // Unsuccessful propagation - should not happen!
  if (!done)
    return done;

  // The z axis is (0,0) - no need to go further
  if (fabs(rz - r0) < 1E-10)
    return done;
  double dist1 = (particle_.X() - x0) * (particle_.X() - x0) + (particle_.Y() - y0) * (particle_.Y() - y0);

  // We are already at closest approach - no need to go further
  if (dist1 < 1E-10)
    return done;

  // Keep for later if it happens to be the right solution
  XYZTLorentzVector vertex1 = particle_.vertex();
  XYZTLorentzVector momentum1 = particle_.momentum();

  // Propagate to the distance of closest approach to (0,0)
  setPropagationConditions(r0, z, first);
  done = backPropagate();
  if (!done)
    return done;

  // Propagate to the first interesection point
  // with cylinder of radius sqrt(x0**2+y0**2)
  setPropagationConditions(rz, z, first);
  done = backPropagate();
  if (!done)
    return done;
  double dist2 = (particle_.X() - x0) * (particle_.X() - x0) + (particle_.Y() - y0) * (particle_.Y() - y0);

  // Keep the good solution.
  if (dist2 > dist1) {
    particle_.setVertex(vertex1);
    particle_.setMomentum(momentum1.X(), momentum1.Y(), momentum1.Z(), momentum1.E());
  }

  // Done
  return done;
}

bool BaseParticlePropagator::propagateToEcal(bool first) {
  //
  // Propagation to Ecal (including preshower) after the
  // last Tracker layer
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder
  //  setPropagationConditions(1290. , 3045 , first);
  //  New position of the preshower as of CMSSW_1_7_X
  setPropagationConditions(129.0, 303.353, first);
  return propagate();
}

bool BaseParticlePropagator::propagateToPreshowerLayer1(bool first) {
  //
  // Propagation to Preshower Layer 1
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder
  //  setPropagationConditions(1290., 3045 , first);
  //  New position of the preshower as of CMSSW_1_7_X
  setPropagationConditions(129.0, 303.353, first);
  bool done = propagate();

  // Check that were are on the Layer 1
  if (done && (particle_.R2() > 125.0 * 125.0 || particle_.R2() < 45.0 * 45.0))
    success = 0;

  return done;
}

bool BaseParticlePropagator::propagateToPreshowerLayer2(bool first) {
  //
  // Propagation to Preshower Layer 2
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder
  //  setPropagationConditions(1290. , 3090 , first);
  //  New position of the preshower as of CMSSW_1_7_X
  setPropagationConditions(129.0, 307.838, first);
  bool done = propagate();

  // Check that we are on Layer 2
  if (done && (particle_.R2() > 125.0 * 125.0 || particle_.R2() < 45.0 * 45.0))
    success = 0;

  return done;
}

bool BaseParticlePropagator::propagateToEcalEntrance(bool first) {
  //
  // Propagation to ECAL entrance
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder
  setPropagationConditions(129.0, 320.9, first);
  bool done = propagate();

  // Go to endcap cylinder in the "barrel cut corner"
  // eta = 1.479 -> cos^2(theta) = 0.81230
  //  if ( done && eta > 1.479 && success == 1 ) {
  if (done && particle_.cos2ThetaV() > 0.81230 && success == 1) {
    setPropagationConditions(152.6, 320.9, first);
    done = propagate();
  }

  // We are not in the ECAL acceptance
  // eta = 3.0 -> cos^2(theta) = 0.99013
  if (particle_.cos2ThetaV() > 0.99014)
    success = 0;

  return done;
}

bool BaseParticlePropagator::propagateToHcalEntrance(bool first) {
  //
  // Propagation to HCAL entrance
  // TODO: include proper geometry
  // Geometry taken from CMSSW_3_1_X xml (Sunanda)
  //

  // First propagate to global barrel / endcap cylinder
  setPropagationConditions(177.5, 335.0, first);
  propDir = 0;
  bool done = propagate();
  propDir = 1;

  // If went through the bottom of HB cylinder -> re-propagate to HE surface
  if (done && success == 2) {
    setPropagationConditions(300.0, 400.458, first);
    propDir = 0;
    done = propagate();
    propDir = 1;
  }

  // out of the HB/HE acceptance
  // eta = 3.0 -> cos^2(theta) = 0.99014
  if (done && particle_.cos2ThetaV() > 0.99014)
    success = 0;

  return done;
}

bool BaseParticlePropagator::propagateToVFcalEntrance(bool first) {
  //
  // Propagation to VFCAL (HF) entrance
  // Geometry taken from xml: Geometry/CMSCommonData/data/cms.xml

  setPropagationConditions(400.0, 1114.95, first);
  propDir = 0;
  bool done = propagate();
  propDir = 1;

  if (!done)
    success = 0;

  // We are not in the VFCAL acceptance
  // eta = 3.0  -> cos^2(theta) = 0.99014
  // eta = 5.2  -> cos^2(theta) = 0.9998755
  double c2teta = particle_.cos2ThetaV();
  if (done && (c2teta < 0.99014 || c2teta > 0.9998755))
    success = 0;

  return done;
}

bool BaseParticlePropagator::propagateToHcalExit(bool first) {
  //
  // Propagation to HCAL exit
  // TODO: include proper geometry
  // Geometry taken from CMSSW_3_1_X xml (Sunanda)
  //

  // Approximate it to a single cylinder as it is not that crucial.
  setPropagationConditions(263.9, 554.1, first);
  //  this->rawPart().particle_.setCharge(0.0); ?? Shower Propagation ??
  propDir = 0;
  bool done = propagate();
  propDir = 1;

  return done;
}

bool BaseParticlePropagator::propagateToHOLayer(bool first) {
  //
  // Propagation to Layer0 of HO (Ring-0)
  // TODO: include proper geometry
  // Geometry taken from CMSSW_3_1_X xml (Sunanda)
  //

  // Approximate it to a single cylinder as it is not that crucial.
  setPropagationConditions(387.6, 1000.0, first);  //Dia is the middle of HO \sqrt{384.8**2+46.2**2}
  //  setPropagationConditions(410.2, 1000.0, first); //sqrt{406.6**2+54.2**2} //for outer layer
  //  this->rawPart().setCharge(0.0); ?? Shower Propagation ??
  propDir = 0;
  bool done = propagate();
  propDir = 1;

  if (done && std::abs(particle_.Z()) > 700.25)
    success = 0;

  return done;
}

bool BaseParticlePropagator::propagateToNominalVertex(const XYZTLorentzVector& v) {
  // Not implemented for neutrals (used for electrons only)
  if (particle_.charge() == 0. || bField == 0.)
    return false;

  // Define the proper pT direction to meet the point (vx,vy) in (x,y)
  double dx = particle_.X() - v.X();
  double dy = particle_.Y() - v.Y();
  double phi = std::atan2(dy, dx) + std::asin(std::sqrt(dx * dx + dy * dy) / (2. * helixRadius()));

  // The absolute pT (kept to the original value)
  double pT = particle_.pt();

  // Set the new pT
  particle_.SetPx(pT * std::cos(phi));
  particle_.SetPy(pT * std::sin(phi));

  return propagateToClosestApproach(v.X(), v.Y());
}

bool BaseParticlePropagator::propagateToBeamCylinder(const XYZTLorentzVector& v, double radius) {
  // For neutral (or BField = 0, simply check that the track passes through the cylinder
  if (particle_.charge() == 0. || bField == 0.)
    return fabs(particle_.Px() * particle_.Y() - particle_.Py() * particle_.X()) / particle_.Pt() < radius;

  // Now go to the charged particles

  // A few needed intermediate variables (to make the code understandable)
  // The inner cylinder
  double r = radius;
  double r2 = r * r;
  double r4 = r2 * r2;
  // The two hits
  double dx = particle_.X() - v.X();
  double dy = particle_.Y() - v.Y();
  double dz = particle_.Z() - v.Z();
  double Sxy = particle_.X() * v.X() + particle_.Y() * v.Y();
  double Dxy = particle_.Y() * v.X() - particle_.X() * v.Y();
  double Dxy2 = Dxy * Dxy;
  double Sxy2 = Sxy * Sxy;
  double SDxy = dx * dx + dy * dy;
  double SSDxy = std::sqrt(SDxy);
  double ATxy = std::atan2(dy, dx);
  // The coefficients of the second order equation for the trajectory radius
  double a = r2 - Dxy2 / SDxy;
  double b = r * (r2 - Sxy);
  double c = r4 - 2. * Sxy * r2 + Sxy2 + Dxy2;

  // Here are the four possible solutions for
  // 1) The trajectory radius
  std::array<double, 4> helixRs = {{0.}};
  helixRs[0] = (b - std::sqrt(b * b - a * c)) / (2. * a);
  helixRs[1] = (b + std::sqrt(b * b - a * c)) / (2. * a);
  helixRs[2] = -helixRs[0];
  helixRs[3] = -helixRs[1];
  // 2) The azimuthal direction at the second point
  std::array<double, 4> helixPhis = {{0.}};
  helixPhis[0] = std::asin(SSDxy / (2. * helixRs[0]));
  helixPhis[1] = std::asin(SSDxy / (2. * helixRs[1]));
  helixPhis[2] = -helixPhis[0];
  helixPhis[3] = -helixPhis[1];
  // Only two solutions are valid though
  double solution1 = 0.;
  double solution2 = 0.;
  double phi1 = 0.;
  double phi2 = 0.;
  // Loop over the four possibilities
  for (unsigned int i = 0; i < 4; ++i) {
    helixPhis[i] = ATxy + helixPhis[i];
    // The centre of the helix
    double xC = helixCentreX(helixRs[i], helixPhis[i]);
    double yC = helixCentreY(helixRs[i], helixPhis[i]);
    // The distance of that centre to the origin
    double dist = helixCentreDistToAxis(xC, yC);
    /*
    std::cout << "Solution "<< i << " = " << helixRs[i] << " accepted ? " 
	      << fabs(fabs(helixRs[i]) - dist ) << " - " << radius << " = " 
	      << fabs(fabs(helixRs[i]) - dist ) - fabs(radius) << " " 
	      << ( fabs( fabs(fabs(helixRs[i]) - dist) -fabs(radius) ) < 1e-6)  << std::endl;
    */
    // Check that the propagation will lead to a trajectroy tangent to the inner cylinder
    if (fabs(fabs(fabs(helixRs[i]) - dist) - fabs(radius)) < 1e-6) {
      // Fill first solution
      if (solution1 == 0.) {
        solution1 = helixRs[i];
        phi1 = helixPhis[i];
        // Fill second solution (ordered)
      } else if (solution2 == 0.) {
        if (helixRs[i] < solution1) {
          solution2 = solution1;
          solution1 = helixRs[i];
          phi2 = phi1;
          phi1 = helixPhis[i];
        } else {
          solution2 = helixRs[i];
          phi2 = helixPhis[i];
        }
        // Must not happen!
      } else {
        std::cout << "warning !!! More than two solutions for this track !!! " << std::endl;
      }
    }
  }

  // Find the largest possible pT compatible with coming from within the inne cylinder
  double pT = 0.;
  double helixPhi = 0.;
  double helixR = 0.;
  // This should not happen
  if (solution1 * solution2 == 0.) {
    std::cout << "warning !!! Less than two solution for this track! " << solution1 << " " << solution2 << std::endl;
    return false;
    // If the two solutions have opposite sign, it means that pT = infinity is a possibility.
  } else if (solution1 * solution2 < 0.) {
    pT = 1000.;
    double norm = pT / SSDxy;
    particle_.setCharge(+1.);
    particle_.setMomentum(dx * norm, dy * norm, dz * norm, 0.);
    particle_.SetE(std::sqrt(particle_.momentum().Vect().Mag2()));
    // Otherwise take the solution that gives the largest transverse momentum
  } else {
    if (solution1 < 0.) {
      helixR = solution1;
      helixPhi = phi1;
      particle_.setCharge(+1.);
    } else {
      helixR = solution2;
      helixPhi = phi2;
      particle_.setCharge(-1.);
    }
    pT = fabs(helixR) * 1e-5 * c_light() * bField;
    double norm = pT / SSDxy;
    particle_.setMomentum(pT * std::cos(helixPhi), pT * std::sin(helixPhi), dz * norm, 0.);
    particle_.SetE(std::sqrt(particle_.momentum().Vect().Mag2()));
  }

  // Propagate to closest approach to get the Z value (a bit of an overkill)
  return propagateToClosestApproach();
}

double BaseParticlePropagator::xyImpactParameter(double x0, double y0) const {
  double ip = 0.;
  double pT = particle_.Pt();

  if (particle_.charge() != 0.0 && bField != 0.0) {
    double radius = helixRadius(pT);
    double phi0 = helixStartPhi();

    // The Helix centre coordinates are computed wrt to the z axis
    double xC = helixCentreX(radius, phi0);
    double yC = helixCentreY(radius, phi0);
    double distz = helixCentreDistToAxis(xC - x0, yC - y0);
    ip = distz - fabs(radius);
  } else {
    ip = fabs(particle_.Px() * (particle_.Y() - y0) - particle_.Py() * (particle_.X() - x0)) / pT;
  }

  return ip;
}
