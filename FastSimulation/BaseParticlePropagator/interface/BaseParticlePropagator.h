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

//CLHEP
#include "CLHEP/Units/PhysicalConstants.h" // for c_light

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
  bool propagateToClosestApproach(bool first=true);
  bool propagateToEcal(bool first=true);
  bool propagateToPreshowerLayer1(bool first=true);
  bool propagateToPreshowerLayer2(bool first=true);
  bool propagateToEcalEntrance(bool first=true);
  bool propagateToHcalEntrance(bool first=true);
  bool propagateToVFcalEntrance(bool first=true);
  bool propagateToHcalExit(bool first=true);
  bool propagateToNominalVertex(
      const HepLorentzVector& hit2=HepLorentzVector(0.,0.,0.,0.));

  /// Return position (hereafter called "vertex") three-vector of particle
  Hep3Vector position() const;
  /// Return momentum four-vector of particle
  HepLorentzVector momentum() const;

  /// Transverse impact parameter
  double xyImpactParameter() const;
  /// Longitudinal impact parameter
  double zImpactParameter() const;

  /// Is the vertex inside the cylinder ? (stricly inside : true) 
  bool           inside() const;
  bool           inside(double rPos) const;
  /// Is the vertex already on the cylinder surface ? 
  bool           onSurface() const;
  bool           onSurface(double rPos) const;
  /// Is the vertex already on the cylinder barrel ? 
  bool           onBarrel() const;
  bool           onBarrel(double rPos) const;
  /// Is the vertex already on the cylinder endcap ? 
  bool           onEndcap() const;
  bool           onEndcap(double rPos) const;
  /// Is the vertex on some material ?
  bool           onFiducial() const { return fiducial; }
  /// Has the particle decayed while propagated ?
  bool           hasDecayed() const { return decayed; }
  /// Has propagation been performed and was barrel or endcap reached ?
  int           getSuccess() const { return success;  }
  /// Set the magnetic field
  inline void setMagneticField(double b) {  bField=b; }
  /// Get the magnetic field 
  inline double getMagneticField() const {  return bField; }
    
  /// Set the propagation characteristics (rCyl, zCyl and first loop only)
  void setPropagationConditions(double r, double z, bool firstLoop=true);

  /// Set the proper decay time
  void setProperDecayTime(double t) { properDecayTime = t; }

  void increaseRCyl(double delta) {rCyl = rCyl + delta;}

private:
  /// Simulated particle that is to be resp has been propagated
  //  RawParticle   particle;
  /// Radius of the cylinder (centred at 0,0,0) to which propagation is done
  double rCyl;
  /// Half-height of the cylinder (centred at 0,0,0) to which propagation is done
  double zCyl;
  /// Magnetic field in the cylinder, oriented along the Z axis
  double bField;
  /// The proper decay time of the particle
  double properDecayTime;

protected:
  /// 0:propagation still be done, 1:reached 'barrel', 2:reached 'endcaps'
  int success;
  /// The particle traverses some real material
  bool fiducial;


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
  /// The helix Radius
  double        helixRadius() const;
  double        helixRadius(double pT) const;
  /// The azimuth of the momentum at the vertex
  double        helixStartPhi() const;
  /// The x coordinate of the helix axis
  double        helixCentreX() const;
  double        helixCentreX(double radius, double phi) const;
  /// The y coordinate of the helix axis
  double        helixCentreY() const;
  double        helixCentreY(double radius, double phi) const;
  /// The distance between the cylinder and the helix axes
  double        helixCentreDistToAxis() const;
  double        helixCentreDistToAxis(double xC, double yC) const;
  /// The azimuth if the vector joining the cylinder and the helix axes
  double        helixCentrePhi() const;
  double        helixCentrePhi(double xC, double yC) const;
};

#endif
