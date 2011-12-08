/*Emacs: -*- C++ -*- */
#ifndef BASEPARTICLEPROPAGATOR_H
#define BASEPARTICLEPROPAGATOR_H

/**
* This class is aimed at propagating charged and neutral 
* particles (yet under the form of a RawParticle) from 
* a given vertex to a cylinder, defined by a radius rCyl 
* and a length 2*zCyl, centered in (0,0,0) and whose axis
* is parallel to the B field (B is oriented along z, by 
* definition of the z axis).
*
* The main method
*
*     bool propagate()
*
* returns true if an intersection between the propagated 
* RawParticle and the cylinder is found. The location of 
* the intersection is given by the value of success:
*
*   * success = 2 : an intersection is found forward in 
*                   the cylinder endcaps. The RawParticle
*                   vertex and momentum are overwritten by
*                   the values at this intersection.
*
*   * success = 1 : an intersection is found forward in 
*                   the cylinder barrel. The RawParticle
*                   vertex and momentum are overwritten by
*                   the values at this intersection.
*
*   * success =-2 : an intersection is found backward in 
*                   the cylinder endcaps. The RawParticle
*                   vertex and momentum are NOT overwritten
*                   by the values at this intersection.
*
*   * success =-1 : an intersection is found backward in 
*                   the cylinder barrel. The RawParticle
*                   vertex and momentum are NOT overwritten
*                   by the values at this intersection.
*
*   * success = 0 : No intersection is found after half a
*                   loop (only possible if firstLoop=true).
*                   The vertex and momentum are overwritten
*                   by the values after this half loop. 
*
* 
* The method
*
*     propagated()
*
* returns a new RawParticle with the propagated coordinates, 
* if overwriting is not considered an advantage by the user.
*
* Particles can be propagated backward with the method
*
*     backPropagate()
*
* Member functions 
*
*  o propagateToPreshowerLayer1(),  
*  o propagateToPreshowerLayer2(),  
*  o propagateToEcalEntrance(),  
*  o propagateToHcalEntrance(),  
*  o propagateToHcalExit(),  
*  o propagateToClosestApproach(),  
*
* only need a momentum, a vertex and an electric charge
* to operate. Radii, half-lengths and default B field (4T)
* are set therein by default.
*
* As of today, no average loss of energy (dE/dx, Brems) is
* considered in the propagation. No uncertainty (multiple
* scattering, dE/dx, Brems) is yet implemented.
*
* \author Patrick Janot 
* $Date : 19-Aug-2002, with subsequent modification for FAMOS
* \version 15-Dec-2003 */

//FAMOS
#include "FastSimulation/Particle/interface/RawParticle.h"

class BaseParticlePropagator : public RawParticle {
  
public:

  /// Default c'tor
  BaseParticlePropagator();

  /** Constructors taking as arguments a RawParticle, as well as the radius,
      half-height and magnetic field defining the cylinder for which 
      propagation is to be performed, and optionally, the proper decay time */
  BaseParticlePropagator(const RawParticle& myPart, 
			 double r, double z, double B);
  BaseParticlePropagator(const RawParticle& myPart, 
			 double r, double z, double B, double t);

  /// Initialize internal switches and quantities
  void init();

  /** Update the current instance, after the propagation of the particle 
      to the surface of the cylinder */
  bool propagate();

  /** Update the current instance, after the back-propagation of the particle 
      to the surface of the cylinder */
  bool backPropagate();

  /** Return a new instance, corresponding to the particle propagated
      to the surface of the cylinder */
  BaseParticlePropagator propagated() const;

  /** Update the particle after propagation to the closest approach from 
      Z axis, to the preshower layer 1 & 2, to the ECAL entrance, to the 
      HCAL entrance, the HCAL 2nd and 3rd layer (not coded yet), the VFCAL 
      entrance, or any BoundSurface(disk or cylinder)*/

  bool propagateToClosestApproach(double x0=0.,double y0=0,bool first=true);
  bool propagateToEcal(bool first=true);
  bool propagateToPreshowerLayer1(bool first=true);
  bool propagateToPreshowerLayer2(bool first=true);
  bool propagateToEcalEntrance(bool first=true);
  bool propagateToHcalEntrance(bool first=true);
  bool propagateToVFcalEntrance(bool first=true);
  bool propagateToHcalExit(bool first=true);
  bool propagateToHOLayer(bool first=true);
  bool propagateToNominalVertex(const XYZTLorentzVector& hit2=XYZTLorentzVector(0.,0.,0.,0.));
  bool propagateToBeamCylinder(const XYZTLorentzVector& v, double radius=0.); 
  /// Set the propagation characteristics (rCyl, zCyl and first loop only)
  void setPropagationConditions(double r, double z, bool firstLoop=true);

private:
  /// Simulated particle that is to be resp has been propagated
  //  RawParticle   particle;
  /// Radius of the cylinder (centred at 0,0,0) to which propagation is done
  double rCyl;
  double rCyl2;
  /// Half-height of the cylinder (centred at 0,0,0) to which propagation is done
  double zCyl;
  /// Magnetic field in the cylinder, oriented along the Z axis
  double bField;
  /// The proper decay time of the particle
  double properDecayTime;
  /// The debug level
  bool debug;

protected:
  /// 0:propagation still be done, 1:reached 'barrel', 2:reached 'endcaps'
  int success;
  /// The particle traverses some real material
  bool fiducial;

  /// The speed of light in mm/ns (!) without clhep (yeaaahhh!)
  inline double c_light() const { return 299.792458; }

private:
  /// Do only the first half-loop
  bool firstLoop;
  /// The particle decayed while propagated !
  bool decayed;
  /// The proper time of the particle
  double properTime;
  /// The propagation direction 
  int propDir;

public:

  /// Set the proper decay time
  inline void setProperDecayTime(double t) { properDecayTime = t; }

  /// Just an internal trick
  inline void increaseRCyl(double delta) {rCyl = rCyl + delta; rCyl2 = rCyl*rCyl; }

  /// Transverse impact parameter
  double xyImpactParameter(double x0=0., double y0=0.) const;

  /// Longitudinal impact parameter
  inline double zImpactParameter(double x0=0, double y0=0.) const {
    // Longitudinal impact parameter
    return Z() - Pz() * std::sqrt( ((X()-x0)*(X()-x0) + (Y()-y0)*(Y()-y0) ) / Perp2());
  }

  /// The helix Radius
  inline double helixRadius() const { 
    // The helix Radius
    //
    // The helix' Radius sign accounts for the orientation of the magnetic field 
    // (+ = along z axis) and the sign of the electric charge of the particle. 
    // It signs the rotation of the (charged) particle around the z axis: 
    // Positive means anti-clockwise, negative means clockwise rotation.
    //
    // The radius is returned in cm to match the units in RawParticle.
    return charge() == 0 ? 0.0 : - Pt() / ( c_light() * 1e-5 * bField * charge() );
  }

  inline double helixRadius(double pT) const { 
    // a faster version of helixRadius, once Perp() has been computed
    return charge() == 0 ? 0.0 : - pT / ( c_light() * 1e-5 * bField * charge() );
  }

  /// The azimuth of the momentum at the vertex
  inline double helixStartPhi() const { 
    // The azimuth of the momentum at the vertex
    return Px() == 0.0 && Py() == 0.0 ? 0.0 : std::atan2(Py(),Px());
  }
  
  /// The x coordinate of the helix axis
  inline double helixCentreX() const { 
    // The x coordinate of the helix axis
    return X() - helixRadius() * std::sin ( helixStartPhi() );
  }

  inline double helixCentreX(double radius, double phi) const { 
    // Fast version of helixCentreX()
    return X() - radius * std::sin (phi);
  }

  /// The y coordinate of the helix axis
  inline double helixCentreY() const { 
    // The y coordinate of the helix axis
    return Y() + helixRadius() * std::cos ( helixStartPhi() );
}

  inline double helixCentreY(double radius, double phi) const { 
    // Fast version of helixCentreX()
    return Y() + radius * std::cos (phi);
  }

  /// The distance between the cylinder and the helix axes
  inline double helixCentreDistToAxis() const { 
    // The distance between the cylinder and the helix axes
    double xC = helixCentreX();
    double yC = helixCentreY();
    return std::sqrt( xC*xC + yC*yC );
  }

  inline double helixCentreDistToAxis(double xC, double yC) const { 
    // Faster version of helixCentreDistToAxis
    return std::sqrt( xC*xC + yC*yC );
  }

  /// The azimuth if the vector joining the cylinder and the helix axes
  inline double helixCentrePhi() const { 
    // The azimuth if the vector joining the cylinder and the helix axes
    double xC = helixCentreX();
    double yC = helixCentreY();
    return xC == 0.0 && yC == 0.0 ? 0.0 : std::atan2(yC,xC);
  }
  
  inline double helixCentrePhi(double xC, double yC) const { 
    // Faster version of helixCentrePhi() 
    return xC == 0.0 && yC == 0.0 ? 0.0 : std::atan2(yC,xC);
  }

  /// Is the vertex inside the cylinder ? (stricly inside : true) 
  inline bool inside() const {
    return (R2()<rCyl2-0.00001*rCyl && fabs(Z())<zCyl-0.00001);}

  inline bool inside(double rPos2) const {
    return (rPos2<rCyl2-0.00001*rCyl && fabs(Z())<zCyl-0.00001);}


  /// Is the vertex already on the cylinder surface ? 
  inline bool onSurface() const {
    return ( onBarrel() || onEndcap() ); 
  }

  inline bool onSurface(double rPos2) const {
    return ( onBarrel(rPos2) || onEndcap(rPos2) ); 
  }

  /// Is the vertex already on the cylinder barrel ? 
  inline bool onBarrel() const {
    double rPos2 = R2();
    return ( fabs(rPos2-rCyl2) < 0.00001*rCyl && fabs(Z()) <= zCyl );
  }

  inline bool onBarrel(double rPos2) const {
    return ( fabs(rPos2-rCyl2) < 0.00001*rCyl && fabs(Z()) <= zCyl );
  }

  /// Is the vertex already on the cylinder endcap ? 
  inline bool onEndcap() const {
    return ( fabs(fabs(Z())-zCyl) < 0.00001 && R2() <= rCyl2 ); 
  }
  
  inline bool onEndcap(double rPos2) const {
    return ( fabs(fabs(Z())-zCyl) < 0.00001 && rPos2 <= rCyl2 ); 
  }

  /// Is the vertex on some material ?
  inline bool onFiducial() const { return fiducial; }

  /// Has the particle decayed while propagated ?
  inline bool hasDecayed() const { return decayed; }

  /// Has propagation been performed and was barrel or endcap reached ?
  inline int  getSuccess() const { return success;  }

  /// Set the magnetic field
  inline void setMagneticField(double b) {  bField=b; }

  /// Get the magnetic field 
  inline double getMagneticField() const {  return bField; }

  /// Set the debug leve;
  inline void setDebug() { debug = true; } 
  inline void resetDebug() { debug = false; }
    
};

#endif
