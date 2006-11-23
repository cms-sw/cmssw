//FAMOS headers
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include <iomanip>
//#include <fstream>
//#include <iostream>

using namespace std;

BaseParticlePropagator::BaseParticlePropagator() :
  RawParticle(), rCyl(0.), zCyl(0.), bField(0)
{;}

BaseParticlePropagator::BaseParticlePropagator( 
         const RawParticle& myPart,double R, double Z, double B ) :
  RawParticle(myPart), rCyl(R), zCyl(Z), bField(B) 
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
  // And this is the proper time at which the particle will decay
  properDecayTime = 
    (pid()==0||pid()==22||abs(pid())==11||abs(pid())==2112||abs(pid())==2212||
     !RandomEngine::instance()) ?
    1E99 : -PDGcTau() * log(RandomEngine::instance()->flatShoot());

}

bool 
BaseParticlePropagator::propagate() { 
  //
  // Check that the particle is not already on the cylinder surface
  //
  double rPos = position().perp();

  if ( onBarrel(rPos) ) { 
    success = 1;
    return true;
  }
  //
  if ( onEndcap(rPos) ) { 
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
    double pT = perp();
    double pT2 = pT*pT;
    double b2 = rCyl * pT;
    double ac = fabs ( x()*py() - y()*px() );
    //
    // The particle never crosses (or never crossed) the cylinder
    //
    if ( ac > b2 ) return false;

    double delta  = sqrt(b2*b2 - ac*ac);
    double tplus  = -( x()*px()+y()*py() ) + delta;
    double tminus = -( x()*px()+y()*py() ) - delta;
    //
    // Find the first (time-wise) intersection point with the cylinder
    //
    double solution = tminus < 0 ? tplus/pT2 : tminus/pT2;
    if ( solution < 0. ) return false;
    //
    // Check that the particle does (did) not exit from the endcaps first.
    //
    double zProp = z() + pz() * solution;
    if ( fabs(zProp) > zCyl ) {
      tplus  = ( zCyl - z() ) / pz();
      tminus = (-zCyl - z() ) / pz();
      solution = tminus < 0 ? tplus : tminus;
      if ( solution < 0. ) return false;
      zProp = z() + pz() * solution;
      success = 2;
    }
    else {
      success = 1;
    }
    //
    // Check the decay time
    //
    double delTime = propDir * (zProp-z())*mass()/pz();
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
    double xProp = x() + px() * solution * factor;
    double yProp = y() + py() * solution * factor;
           zProp = z() + pz() * solution * factor;
    double tProp = t() + e()  * solution * factor;
  //
  // ... and update the particle with its propagated coordinates
  //
    setVertex( HepLorentzVector(xProp,yProp,zProp,tProp) );

    return true;
  }
  //
  // Treat charged particles with magnetic field next.
  //
  else {
    //
    // First check that the particle can cross the cylinder
    //
    double pT = perp();
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
    if ( z() * pz() > zCyl * fabs(pz()) ) { 
      success = -2;
      return true;
    }
    //
    // The particle is already on the barrel and needs not be propgated again
    //
    // if ( vertex().perp() == rCyl ) return false;
    //
    //    double pT = perp();
    double pZ = pz();
    double phiProp, zProp;
    //
    // Does the helix cross the cylinder barrel ? If not, do the first half-loop
    //
    double rProp = min(fabs(radius)+dist-0.000001, rCyl);
    //
    // Compute phi after propagation (two solutions, but the asin definition
    // (between -pi/2 and +pi/2) takes care of the wrong one.
    //
    phiProp = helixCentrePhi(xC,yC) + ( inside(rPos) || onSurface(rPos) ? 
	    asin ( (rProp*rProp-radius*radius-dist*dist) / (2*dist*radius) ) :
       M_PI-asin ( (rProp*rProp-radius*radius-dist*dist) / (2*dist*radius) ) );
    //
    // Solve the obvious two-pi ambiguities: more than one turn!
    //
    if ( abs(phiProp - phi0) > 2.*M_PI )
      phiProp += phiProp > phi0 ? -2.*M_PI : +2.*M_PI;
    //
    // Check that the phi difference has the right sign, according to Q*B
    // (Another two pi ambuiguity, slightly more subtle)
    //
    if ( (phiProp-phi0)*radius < 0.0 )
      radius > 0.0 ? phiProp += 2 * M_PI : phiProp -= 2 * M_PI;
    //
    // Check that the particle does not exit from the endcaps first.
    //
    zProp = z() + ( phiProp - phi0 ) * pZ * radius / pT ;
    if ( fabs(zProp) > zCyl ) {

      zProp = zCyl * fabs(pZ)/pZ;
      phiProp = phi0 + (zProp - z()) / radius * pT / pZ;
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
	  phiProp = phi0 + (zProp - z()) / radius * pT / pZ;
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
    double delTime = propDir * (zProp-z())*mass()/pz();
    double factor = 1.;
    properTime += delTime;
    if ( properTime > properDecayTime ) {
      factor = 1.-(properTime-properDecayTime)/delTime;
      properTime = properDecayTime;
      decayed = true;
    }

    zProp = z() + (zProp-z())*factor;
    phiProp = phi0 + (zProp - z()) / radius * pT / pZ;

    double sProp = sin(phiProp);
    double cProp = cos(phiProp);
    double xProp = xC + radius * sProp;
    double yProp = yC - radius * cProp;
    double tProp = t() + (zProp-z())*e()/pz();
    double pxProp = pT * cProp;
    double pyProp = pT * sProp;
    //
    // Last check : Although propagated to the endcaps, R could still be
    // larger than rCyl because the cylinder was too short...
    //
    if ( success == 2 && sqrt(xProp*xProp+yProp*yProp) > rCyl ) { 
      success = -1;
      return true;
    }
    //
    // ... and update the particle vertex and momentum
    //
    setVertex( HepLorentzVector(xProp,yProp,zProp,tProp) );
    setPx(pxProp);
    setPy(pyProp);
    return true;
 
  }
}

bool
BaseParticlePropagator::backPropagate() {

  // Backpropagate
  setPx(-px()); setPy(-py()); setPz(-pz()); setCharge(-charge());
  propDir = -1;
  bool done = propagate();
  setPx(-px()); setPy(-py()); setPz(-pz()); setCharge(-charge());
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
    r = fabs( px() * y() - py() * x() ) / perp(); 
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
  if ( done && (vertex().perp() > 125.0 || vertex().perp() < 45.0) ) 
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
  if ( done && (vertex().perp() > 125.0 || vertex().perp() < 45.0 ))
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

  // Where are we ?
  double eta = fabs ( position().eta() );

  // Go to endcap cylinder in the "barrel cut corner" 
  if ( done && eta > 1.479 && success == 1 ) {
    setPropagationConditions(171.1 , 317.0, first);
    done = propagate();
  }

  // We are not in the ECAL acceptance
  if ( fabs ( position().eta() ) > 3.0 ) success = 0;

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
  if ( done && fabs ( position().eta() ) > 3.0 ) success = 0;

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
  double feta = fabs ( position().eta() );
  if ( done && ( feta < 3 || feta > 5.0 ) ) success = 0;

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
BaseParticlePropagator::propagateToNominalVertex(const HepLorentzVector& v) 
{

  // Not implemented for neutrals (used for electrons only)
  if ( charge() == 0. || bField == 0.) return false;

  // Define the proper pT direction to meet the point (vx,vy) in (x,y)
  double dx = x()-v.x();
  double dy = y()-v.y();
  double phi = atan2(dy,dx) + asin ( sqrt(dx*dx+dy*dy)/(2.*helixRadius()) );

  // The absolute pT (kept to the original value)
  double pT = perp();

  // Set the new pT
  setPx(pT*cos(phi));
  setPy(pT*sin(phi));

  return propagateToClosestApproach();
  
}

double
BaseParticlePropagator::xyImpactParameter() const {
  // Transverse impact parameter
  return ( charge() != 0.0 && bField != 0.0 ) ? 
    helixCentreDistToAxis() - fabs( helixRadius() ) :
    fabs( px() * y() - py() * x() ) / perp(); 
}

double
BaseParticlePropagator::zImpactParameter() const {
  // Longitudinal impact parameter
  return vertex().z() - vertex().perp() * pz() / perp();
}

double 
BaseParticlePropagator::helixRadius() const { 
  // The helix Radius
  //
  // The helix' Radius sign accounts for the orientation of the magnetic field 
  // (+ = along z axis) and the sign of the electric charge of the particle. 
  // It signs the rotation of the (charged) particle around the z axis: 
  // Positive means anti-clockwise, negative means clockwise rotation.
  //
  // The radius is returned in cm to match the units in RawParticle.
  return charge() == 0 ? 0.0 : - perp() / ( c_light * 1e-5 * bField * charge() );
}

double 
BaseParticlePropagator::helixRadius(double pT) const { 
  // a faster version of helixRadius, once perp() has been computed
  return charge() == 0 ? 0.0 : - pT / ( c_light * 1e-5 * bField * charge() );
}

double 
BaseParticlePropagator::helixStartPhi() const { 
  // The azimuth of the momentum at the vertex
  return px() == 0.0 && py() == 0.0 ? 0.0 : atan2(py(),px());
}

double 
BaseParticlePropagator::helixCentreX() const { 
  // The x coordinate of the helix axis
  return x() - helixRadius() * sin ( helixStartPhi() );
}

double 
BaseParticlePropagator::helixCentreY() const { 
  // The y coordinate of the helix axis
  return y() + helixRadius() * cos ( helixStartPhi() );
}

double 
BaseParticlePropagator::helixCentreX(double radius, double phi) const { 
  // Fast version of helixCentreX()
  return x() - radius * sin (phi);
}

double 
BaseParticlePropagator::helixCentreY(double radius, double phi) const { 
  // Fast version of helixCentreX()
  return y() + radius * cos (phi);
}

double 
BaseParticlePropagator::helixCentreDistToAxis() const { 
  // The distance between the cylinder and the helix axes
  double xC = helixCentreX();
  double yC = helixCentreY();
  return sqrt( xC*xC + yC*yC );
}

double 
BaseParticlePropagator::helixCentreDistToAxis(double xC, double yC) const { 
  // Faster version of helixCentreDistToAxis
  //  double xC = helixCentreX();
  //  double yC = helixCentreY();
  return sqrt( xC*xC + yC*yC );
}

double 
BaseParticlePropagator::helixCentrePhi() const { 
  // The azimuth if the vector joining the cylinder and the helix axes
  double xC = helixCentreX();
  double yC = helixCentreY();
  return xC == 0.0 && yC == 0.0 ? 0.0 : atan2(yC,xC);
}

double 
BaseParticlePropagator::helixCentrePhi(double xC, double yC) const { 
  // Faster version of helixCentrePhi() 
  //  double xC = helixCentreX();
  //  double yC = helixCentreY();
  return xC == 0.0 && yC == 0.0 ? 0.0 : atan2(yC,xC);
}

Hep3Vector
BaseParticlePropagator::position() const { return vertex().vect(); }

HepLorentzVector
BaseParticlePropagator::momentum() const { 
  return HepLorentzVector(px(),py(),pz(),e()); 
}

bool
BaseParticlePropagator::inside() const {
  return (position().perp()<rCyl-0.00001 && fabs(z())<zCyl-0.00001);}

bool BaseParticlePropagator::onSurface() const {
  return ( onBarrel() || onEndcap() ); 
}

bool BaseParticlePropagator::onBarrel() const {
  return ( fabs(position().perp()-rCyl) < 0.00001 && fabs(z()) <= zCyl );
}

bool BaseParticlePropagator::onEndcap() const {
  return ( fabs(fabs(z())-zCyl) < 0.00001 && position().perp() <= rCyl ); 
}

// Faster versions
bool
BaseParticlePropagator::inside(double rPos) const {
  return (rPos<rCyl-0.00001 && fabs(z())<zCyl-0.00001);}

bool BaseParticlePropagator::onSurface(double rPos) const {
  return ( onBarrel(rPos) || onEndcap(rPos) ); 
}

bool BaseParticlePropagator::onBarrel(double rPos) const {
  return ( fabs(rPos-rCyl) < 0.00001 && fabs(z()) <= zCyl );
}

bool BaseParticlePropagator::onEndcap(double rPos) const {
  return ( fabs(fabs(z())-zCyl) < 0.00001 && rPos <= rCyl ); 
}


