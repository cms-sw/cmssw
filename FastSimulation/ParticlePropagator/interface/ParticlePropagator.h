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

// FAMOS Headers
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

class TrackerLayer;
class FSimTrack;
class RandomEngine;
class MagneticFieldMap;

class ParticlePropagator : public BaseParticlePropagator {
  
public:
  /// Default c'tor
  ParticlePropagator();

  /** Constructor taking as arguments a RawParticle, as well as the radius,
      half-height and magnetic field defining the cylinder for which 
      propagation is to be performed */
  ParticlePropagator(const RawParticle& myPart, 
		     double R, double Z,
		     const MagneticFieldMap* aFieldMap,
		     const RandomEngine* engine);
  
  /** Constructor with only a RawParticle as argument for subsequent 
      propagation to known surfaces (ECAL, HCAL ...) */
  ParticlePropagator(const RawParticle& myPart,
		     const MagneticFieldMap* aFieldMap,
		     const RandomEngine* engine);

  /** Constructor with two LorentzVector (momentum and vertex (in cm)) and 
      an electric charge propagation to known surfaces (ECAL, HCAL ...) */
  ParticlePropagator(const XYZTLorentzVector& p, 
		     const XYZTLorentzVector& v, 
		     float q,
		     const MagneticFieldMap* aFieldMap);

  /** Constructor with a LorentzVector (momentum), a Hep3Vector (vertex in cm)
      and an electric charge propagation to known surfaces (ECAL, HCAL ...) */
  ParticlePropagator(const XYZTLorentzVector& p, 
		     const XYZVector& v, float q,
		     const MagneticFieldMap* aFieldMap);

  /** Constructor with a FSimTrack from the FSimEvent*/
  ParticlePropagator(const FSimTrack& simTrack,
		     const MagneticFieldMap* aFieldMap,
		     const RandomEngine* engine);

  /** Constructor with a (Base)ParticlePropagator*/
  ParticlePropagator(const ParticlePropagator& myPropPart);
  //  ParticlePropagator(BaseParticlePropagator myPropPart);
  ParticlePropagator(const BaseParticlePropagator &myPropPart,
		     const MagneticFieldMap* aFieldMap);

  /**Initialize the proper decay time of the particle*/
  void initProperDecayTime();

  /** Return a new instance, corresponding to the particle propagated
      to the surface of the cylinder */
  ParticlePropagator propagated() const;

  /** Update the particle after propagation to the closest approach from 
      Z axis, to the preshower layer 1 & 2, to the ECAL entrance, to the 
      HCAL entrance, the HCAL 2nd and 3rd layer (not coded yet), the VFCAL 
      entrance, or any BoundSurface(disk or cylinder)*/
  bool propagateToClosestApproach(double x0=0., double y0=0., bool first=true);
  bool propagateToNominalVertex(const XYZTLorentzVector& hit2=
			              XYZTLorentzVector(0.,0.,0.,0.));

  /** The fieldMap given by the detector geormetry */
  double fieldMap(double x,double y,double z);
  double fieldMap(const TrackerLayer& layer, double coord, int success);
 
  bool propagateToBoundSurface(const TrackerLayer&);
  void setPropagationConditions(const TrackerLayer&, 
				bool firstLoop=true);

private:

  const MagneticFieldMap* theFieldMap;
  const RandomEngine* random;

};

#endif
