//FAMOS headers
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

BaseParticlePropagator::BaseParticlePropagator() :
  RawParticle(), rCyl(0.), rCyl2(0.), zCyl(0.), bField(0), properDecayTime(1E99)
{;}

BaseParticlePropagator::BaseParticlePropagator( 
         const RawParticle& myPart,double R, double Z, double B, double t ) :
  RawParticle(myPart), rCyl(R), rCyl2(R*R), zCyl(Z), bField(B), properDecayTime(t) 
{
  init();
}

BaseParticlePropagator::BaseParticlePropagator( 
         const RawParticle& myPart,double R, double Z, double B) :
  RawParticle(myPart), rCyl(R), rCyl2(R*R), zCyl(Z), bField(B), properDecayTime(1E99) 
{
  init();
}

void 
BaseParticlePropagator::init() {
  
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

}

bool 
BaseParticlePropagator::propagate() { 
  //
  // Check that the particle is not already on the cylinder surface
  //
  double rPos2 = R2();

  if ( onBarrel(rPos2) ) { 
    success = 1;
    return true;
  }
  //
  if ( onEndcap(rPos2) ) { 
    success = 2;
    return true;
  }
  //
  // Treat neutral particles (or no magnetic field) first 
  //
  if ( charge() == 0.0 || bField == 0.0 ) {
    //
    // First check that the particle crosses the cylinder
    //
    double pT2 = Perp2();
    //    double pT2 = pT*pT;
    //    double b2 = rCyl * pT;
    double b2 = rCyl2 * pT2;
    double ac = fabs ( X()*Py() - Y()*Px() );
    double ac2 = ac*ac;
    //
    // The particle never crosses (or never crossed) the cylinder
    //
    //    if ( ac > b2 ) return false;
    if ( ac2 > b2 ) return false;

    //    double delta  = std::sqrt(b2*b2 - ac*ac);
    double delta  = std::sqrt(b2 - ac2);
    double tplus  = -( X()*Px()+Y()*Py() ) + delta;
    double tminus = -( X()*Px()+Y()*Py() ) - delta;
    //
    // Find the first (time-wise) intersection point with the cylinder
    //
    double solution = tminus < 0 ? tplus/pT2 : tminus/pT2;
    if ( solution < 0. ) return false;
    //
    // Check that the particle does (did) not exit from the endcaps first.
    //
    double zProp = Z() + Pz() * solution;
    if ( fabs(zProp) > zCyl ) {
      tplus  = ( zCyl - Z() ) / Pz();
      tminus = (-zCyl - Z() ) / Pz();
      solution = tminus < 0 ? tplus : tminus;
      if ( solution < 0. ) return false;
      zProp = Z() + Pz() * solution;
      success = 2;
    }
    else {
      success = 1;
    }
    //
    // Check the decay time
    //
    double delTime = propDir * (zProp-Z())*mass()/Pz();
    double factor = 1.;
    properTime += delTime;
    if ( properTime > properDecayTime ) {
      factor = 1.-(properTime-properDecayTime)/delTime;
      properTime = properDecayTime;
      decayed = true;
    }

    //
    // Compute the coordinates of the RawParticle after propagation
    //
    double xProp = X() + Px() * solution * factor;
    double yProp = Y() + Py() * solution * factor;
           zProp = Z() + Pz() * solution * factor;
    double tProp = T() + E()  * solution * factor;
  //
  // ... and update the particle with its propagated coordinates
  //
    setVertex( XYZTLorentzVector(xProp,yProp,zProp,tProp) );

    return true;
  }
  //
  // Treat charged particles with magnetic field next.
  //
  else {
    //
    // First check that the particle can cross the cylinder
    //
    double pT = Pt();
    double radius = helixRadius(pT);
    double phi0 = helixStartPhi();
    double xC = helixCentreX(radius,phi0);
    double yC = helixCentreY(radius,phi0);
    double dist = helixCentreDistToAxis(xC,yC);
    //
    // The helix either is outside or includes the cylinder -> no intersection
    //
    if ( fabs ( fabs(radius) - dist ) > rCyl ) return false;
    //
    // The particle is already away from the endcaps, and will never come back
    // Could be back-propagated, so warn the user that it is so.
    // 
    if ( Z() * Pz() > zCyl * fabs(Pz()) ) { 
      success = -2;
      return true;
    }

    double pZ = Pz();
    double phiProp, zProp;
    //
    // Does the helix cross the cylinder barrel ? If not, do the first half-loop
    //
    double rProp = std::min(fabs(radius)+dist-0.000001, rCyl);

    // Check for rounding errors in the ArcSin.
    double sinPhiProp = 
      (rProp*rProp-radius*radius-dist*dist)/( 2.*dist*radius);
    //
    phiProp =  fabs(sinPhiProp) < 0.99999999  ? 
      std::asin(sinPhiProp) : M_PI/2. * sinPhiProp;

    //
    // Compute phi after propagation (two solutions, but the asin definition
    // (between -pi/2 and +pi/2) takes care of the wrong one.
    //
    phiProp = helixCentrePhi(xC,yC) + 
      ( inside(rPos2) || onSurface(rPos2) ? phiProp : M_PI-phiProp ); 

    //
    // Solve the obvious two-pi ambiguities: more than one turn!
    //
    if ( fabs(phiProp - phi0) > 2.*M_PI )
      phiProp += ( phiProp > phi0 ? -2.*M_PI : +2.*M_PI );

    //
    // Check that the phi difference has the right sign, according to Q*B
    // (Another two pi ambuiguity, slightly more subtle)
    //
    if ( (phiProp-phi0)*radius < 0.0 )
      radius > 0.0 ? phiProp += 2 * M_PI : phiProp -= 2 * M_PI;

    //
    // Check that the particle does not exit from the endcaps first.
    //
    zProp = Z() + ( phiProp - phi0 ) * pZ * radius / pT ;
    if ( fabs(zProp) > zCyl ) {

      zProp = zCyl * fabs(pZ)/pZ;
      phiProp = phi0 + (zProp - Z()) / radius * pT / pZ;
      success = 2;

    } else {

      //
      // If the particle does not cross the barrel, either process the
      // first-half loop only or propagate all the way down to the endcaps
      //
      if ( rProp < rCyl ) {

	if ( firstLoop ) {

	  success = 0;

	} else {

	  zProp = zCyl * fabs(pZ)/pZ;
	  phiProp = phi0 + (zProp - Z()) / radius * pT / pZ;
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
    double delTime = propDir * (zProp-Z())*mass() / pZ;
    double factor = 1.;
    properTime += delTime;
    if ( properTime > properDecayTime ) {
      factor = 1.-(properTime-properDecayTime)/delTime;
      properTime = properDecayTime;
      decayed = true;
    }

    zProp = Z() + (zProp-Z())*factor;

    phiProp = phi0 + (zProp - Z()) / radius * pT / pZ;

    double sProp = std::sin(phiProp);
    double cProp = std::cos(phiProp);
    double xProp = xC + radius * sProp;
    double yProp = yC - radius * cProp;
    double tProp = T() + (zProp-Z())*E()/pZ;
    double pxProp = pT * cProp;
    double pyProp = pT * sProp;

    //
    // Last check : Although propagated to the endcaps, R could still be
    // larger than rCyl because the cylinder was too short...
    //
    if ( success == 2 && xProp*xProp+yProp*yProp > rCyl2 ) { 
      success = -1;
      return true;
    }
    //
    // ... and update the particle vertex and momentum
    //
    setVertex( XYZTLorentzVector(xProp,yProp,zProp,tProp) );
    SetXYZT(pxProp,pyProp,pZ,E());
    //    SetPx(pxProp);
    //    SetPy(pyProp);
    return true;
 
  }
}

bool
BaseParticlePropagator::backPropagate() {

  // Backpropagate
  SetXYZT(-Px(),-Py(),-Pz(),E());
  setCharge(-charge());
  propDir = -1;
  bool done = propagate();
  SetXYZT(-Px(),-Py(),-Pz(),E());
  setCharge(-charge());
  propDir = +1;

  return done;
}

BaseParticlePropagator
BaseParticlePropagator::propagated() const {
  // Copy the input particle in a new instance
  BaseParticlePropagator myPropPart(*this);
  // Allow for many loops
  myPropPart.firstLoop=false;
  // Propagate the particle ! 
  myPropPart.propagate();
  // Back propagate if the forward propagation was not a success
  if (myPropPart.success < 0 )
    myPropPart.backPropagate();
  // Return the new instance
  return myPropPart;
}

void 
BaseParticlePropagator::setPropagationConditions(double R, double Z, bool f) { 
  rCyl = R;
  rCyl2 = R*R;
  zCyl = Z;
  firstLoop = f;
}

bool
BaseParticlePropagator::propagateToClosestApproach(bool first) {
  //
  // Propagation to point of clostest approach to z axis
  //
  double r,z;
  if ( charge() != 0.0 && bField != 0.0 ) {
    r = fabs ( fabs( helixRadius() )
	       - helixCentreDistToAxis() ) + 0.0000001;
  }
  else {
    r = fabs( Px() * Y() - Py() * X() ) / Pt(); 
  }

  z = 999999.;

  setPropagationConditions(r , z, first);

  return backPropagate();

}

bool
BaseParticlePropagator::propagateToEcal(bool first) {
  //
  // Propagation to Ecal (including preshower) after the
  // last Tracker layer
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder 
  //  setPropagationConditions(1290. , 3045 , first);
  setPropagationConditions(129.0 , 303.16 , first);
  return propagate();

}

bool
BaseParticlePropagator::propagateToPreshowerLayer1(bool first) {
  //
  // Propagation to Preshower Layer 1
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder 
  //  setPropagationConditions(1290., 3045 , first);
  setPropagationConditions(129.0, 303.16 , first);
  bool done = propagate();

  // Check that were are on the Layer 1 
  if ( done && (R2() > 125.0*125.0 || R2() < 45.0*45.0) ) 
    success = 0;
  
  return done;
}

bool
BaseParticlePropagator::propagateToPreshowerLayer2(bool first) {
  //
  // Propagation to Preshower Layer 2
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder 
  //  setPropagationConditions(1290. , 3090 , first);
  setPropagationConditions(129.0 , 307.13 , first);
  bool done = propagate();

  // Check that we are on Layer 2 
  if ( done && (R2() > 125.0*125.0 || R2() < 45.0*45.0 ) )
    success = 0;

  return done;

}

bool
BaseParticlePropagator::propagateToEcalEntrance(bool first) {
  //
  // Propagation to ECAL entrance
  // TODO: include proper geometry
  // Geometry taken from CMS ECAL TDR
  //
  // First propagate to global barrel / endcap cylinder 
  setPropagationConditions(129.0 , 317.0,first);
  bool done = propagate();

  // Go to endcap cylinder in the "barrel cut corner" 
  // eta = 1.479 -> cos^2(theta) = 0.81230
  //  if ( done && eta > 1.479 && success == 1 ) {
  if ( done && cos2ThetaV() > 0.81230 && success == 1 ) {
    setPropagationConditions(171.1 , 317.0, first);
    done = propagate();
  }

  // We are not in the ECAL acceptance
  // eta = 3.0 -> cos^2(theta) = 0.99013
  if ( cos2ThetaV() > 0.99013 ) success = 0;

  return done;
}

bool
BaseParticlePropagator::propagateToHcalEntrance(bool first) {
  //
  // Propagation to HCAL entrance
  // TODO: include proper geometry
  // Geometry taken from DAQ TDR Chapter 13
  //

  // First propagate to global barrel / endcap cylinder 
  setPropagationConditions(183.0 , 388.0, first);
  propDir = 0;
  bool done = propagate();
  propDir = 1;

  // We are not in the HCAL acceptance
  // eta = 3.0 -> cos^2(theta) = 0.99013
  if ( done && cos2ThetaV() > 0.99013 ) success = 0;

  return done;
}

bool
BaseParticlePropagator::propagateToVFcalEntrance(bool first) {
  //
  // Propagation to VFCAL entrance
  // TODO: include proper geometry
  // Geometry taken from DAQ TDR Chapter 13

  setPropagationConditions(200.0 , 1110.0, first);
  propDir = 0;
  bool done = propagate();
  propDir = 1;

  // We are not in the VFCAL acceptance
  // eta = 3.0 -> cos^2(theta) = 0.99013
  // eta = 5.0 -> cos^2(theta) = 0.998184
  double c2teta = cos2ThetaV();
  if ( done && ( c2teta < 0.99013 || c2teta > 0.998184 ) ) success = 0;

  return done;
}

bool
BaseParticlePropagator::propagateToHcalExit(bool first) {
  //
  // Propagation to HCAL exit
  // TODO: include proper geometry
  // Geometry taken from DAQ TDR Chapter 13
  //

  // Approximate it to a single cylinder as it is not that crucial.
  setPropagationConditions(285.0 , 560.0, first);
  //  this->rawPart().setCharge(0.0); ?? Shower Propagation ??
  propDir = 0;
  bool done = propagate();
  propDir = 1;

  return done;
}

bool
BaseParticlePropagator::propagateToNominalVertex(const XYZTLorentzVector& v) 
{

  // Not implemented for neutrals (used for electrons only)
  if ( charge() == 0. || bField == 0.) return false;

  // Define the proper pT direction to meet the point (vx,vy) in (x,y)
  double dx = X()-v.X();
  double dy = Y()-v.Y();
  double phi = std::atan2(dy,dx) + std::asin ( std::sqrt(dx*dx+dy*dy)/(2.*helixRadius()) );

  // The absolute pT (kept to the original value)
  double pT = pt();

  // Set the new pT
  SetPx(pT*std::cos(phi));
  SetPy(pT*std::sin(phi));

  return propagateToClosestApproach();
  
}

