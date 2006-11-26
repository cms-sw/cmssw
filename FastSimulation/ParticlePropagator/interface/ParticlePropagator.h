/*Emacs: -*- C++ -*- */
#ifndef PARTICLEPROPAGATOR_H
#define PARTICLEPROPAGATOR_H

/**
* This class extend the BaseParticlePropagator class to the official 
* field map, and with a few additional constructors specific to the 
* framework. The member function 
*
*    propagateToBoundSurface(const Layer&)
*
* allows particles to be propagated to a boundSurface (only
* cylinders and disks, though).
*
* \author Patrick Janot 
* $Date : 19-Aug-2002, with subsequent modification for FAMOS
* \version 15-Dec-2003 */

//CLHEP
#include "CLHEP/Units/PhysicalConstants.h" // for c_light

// FAMOS Headers
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

class TrackerLayer;
class FSimTrack;

class ParticlePropagator : public BaseParticlePropagator {
  
public:
  /// Default c'tor
  ParticlePropagator();

  /** Constructor taking as arguments a RawParticle, as well as the radius,
      half-height and magnetic field defining the cylinder for which 
      propagation is to be performed */
  ParticlePropagator(const RawParticle& myPart, double R, double Z, double B);
  
  /** Constructor with only a RawParticle as argument for subsequent 
      propagation to known surfaces (ECAL, HCAL ...) */
  ParticlePropagator(const RawParticle& myPart);

  /** Constructor with two HepLorentzVector (momentum and vertex (in cm)) and 
      an electric charge propagation to known surfaces (ECAL, HCAL ...) */
  ParticlePropagator(const HepLorentzVector& p, 
		     const HepLorentzVector& v, float q);

  /** Constructor with a HepLorentzVector (momentum), a Hep3Vector (vertex in cm)
      and an electric charge propagation to known surfaces (ECAL, HCAL ...) */
  ParticlePropagator(const HepLorentzVector& p, 
		     const Hep3Vector& v, float q);

  /** Constructor with a FSimTrack from the FSimEvent*/
  ParticlePropagator(const FSimTrack& simTrack);

  /** Constructor with a BaseParticlePropagator*/
  ParticlePropagator(ParticlePropagator& myPropPart);

  /** Return a new instance, corresponding to the particle propagated
      to the surface of the cylinder */
  ParticlePropagator propagated() const;

  /** Update the particle after propagation to the closest approach from 
      Z axis, to the preshower layer 1 & 2, to the ECAL entrance, to the 
      HCAL entrance, the HCAL 2nd and 3rd layer (not coded yet), the VFCAL 
      entrance, or any BoundSurface(disk or cylinder)*/
  bool propagateToClosestApproach(bool first=true);
  bool propagateToNominalVertex(const HepLorentzVector& hit2=
				HepLorentzVector(0.,0.,0.,0.));

  /** The fieldMap given by the detector geormetry */
  double fieldMap(double x,double y,double z);
  double fieldMap(const TrackerLayer& layer, double coord, int success);
 
  bool propagateToBoundSurface(const TrackerLayer&);
  void setPropagationConditions(const TrackerLayer&, 
				bool firstLoop=true);

};

#endif
