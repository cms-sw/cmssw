#ifndef RecoLocalMuon_DTLinearDriftAlgo_H
#define RecoLocalMuon_DTLinearDriftAlgo_H

/** \class DTLinearDriftAlgo
 *  Concrete implementation of DTRecHitBaseAlgo.
 *  Compute drift distance using constant drift velocity
 *  as defined in driftVelocity parameter.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"



class DTLinearDriftAlgo : public DTRecHitBaseAlgo {
 public:
  /// Constructor
  DTLinearDriftAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~DTLinearDriftAlgo();

  // Operations

  /// Whether the algorithm can update hits once the 2D segment is
  /// known (i.e. compute() is implemented for the 2nd step)
  virtual bool canUpdate2D() { //FIXME: Is it really needed?
    return true;
  }


  /// Whether the algorithm can update hits once the 4D segment is
  /// known (i.e. compute() is implemented for the 3rd step)
  virtual bool canUpdate4D() { //FIXME: Is it really needed?
    return true;
  }
    
  /// First step in computation of Left/Right hits from a Digi.  
  /// The results are the local position (in MuBarLayer frame) of the
  /// Left and Right hit, and the error (which is common). Returns
  /// false on failure. 
  virtual bool compute(const DTLayer* layer,
                       const DTDigi& digi,
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const;


  /// Second step in hit position computation, for algorithms which support it.
  /// The impact angle is given as input, and it's used to improve the hit
  /// position (and relative error). The angle is defined in radians, with
  /// respect to the perpendicular to the layer plane, directed outward in CMS
  /// (so toward -z in DTLayer reference frame). Given the local direction,
  /// angle=atan(dir.x()/-dir.z()) . This can be used when a SL segment is
  /// built, so the impact angle is known but the position along wire is not.
  virtual bool compute(const DTLayer* layer,
                       const DTDigi& digi,
                       const float& angle,
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const;


  /// Third (and final) step in hits position computation, for
  /// algorithms which support it.
  /// In addition the the angle, also the global position of the hit is given
  /// as input. This allows to get the magnetic field at the hit position (and
  /// not only that at the center of the wire). Also the position along the
  /// wire is available and can be used to correct the drift time for particle
  /// TOF and propagation of signal along the wire. 
  virtual bool compute(const DTLayer* layer,
                       const DTDigi& digi,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const;


 private:
  // The Drift Velocity (cm/ns)
  static float vDrift;

  // The resolution on the Hits (cm)
  static float hitResolution;

  // Times below MinTime (ns) are considered as coming from previous BXs.
  static float minTime;

  // Times above MaxTime (ns) are considered as coming from following BXs
  static float maxTime;

  // Switch on/off the verbosity
  static bool debug;
};
#endif


