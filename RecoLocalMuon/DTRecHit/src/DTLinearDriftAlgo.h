#ifndef RecoLocalMuon_DTLinearDriftAlgo_H
#define RecoLocalMuon_DTLinearDriftAlgo_H

/** \class DTLinearDriftAlgo
 *  Concrete implementation of DTRecHitBaseAlgo.
 *  Compute drift distance using constant drift velocity
 *  as defined in the "driftVelocity" parameter.
 *
 *  $Date: 2006/03/23 15:39:30 $
 *  $Revision: 1.3 $
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

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);


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
  /// The results are the local position (in DTLayer frame) of the
  /// Left and Right hit, and the error (which is common). Returns
  /// false on failure. The hit is assumed to be at the wire center.
  virtual bool compute(const DTLayer* layer,
                       const DTDigi& digi,
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const;


  /// Second step in hit position computation.
  /// It is the same as first step since the angular information is not used
  virtual bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const;


  /// Third (and final) step in hits position computation.
  /// Also the hit position along the wire is available
  /// and can be used to correct the drift time for particle
  /// TOF and propagation of signal along the wire. 
  virtual bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const;


 private:

  // Do the actual work.
  virtual bool compute(const DTLayer* layer,
		       const DTWireId& wireId,
		       const float digiTime,
		       const GlobalPoint& globPos, 
		       LocalPoint& leftPoint,
		       LocalPoint& rightPoint,
		       LocalError& error,
		       int step) const;

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


