#ifndef RecoLocalMuon_DTLinearDriftAlgo_H
#define RecoLocalMuon_DTLinearDriftAlgo_H

/** \class DTLinearDriftAlgo
 *  Concrete implementation of DTRecHitBaseAlgo.
 *  Compute drift distance using constant drift velocity
 *  as defined in the "driftVelocity" parameter.
 *
 *  $Date: 2007/04/19 11:08:17 $
 *  $Revision: 1.1 $
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
  /// NOTE: Only position and error of the new hit are modified
  virtual bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       DTRecHit1D& newHit1D) const;


  /// Third (and final) step in hits position computation.
  /// Also the hit position along the wire is available
  /// and can be used to correct the drift time for particle
  /// TOF and propagation of signal along the wire. 
  /// NOTE: Only position and error of the new hit are modified
  virtual bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       DTRecHit1D& newHit1D) const;


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

  // Interface to the method which does the actual work suited for 2nd and 3rd steps 
  virtual bool compute(const DTLayer* layer,
		       const DTWireId& wireId,
		       const float digiTime,
		       const GlobalPoint& globPos, 
		       DTRecHit1D& newHit1D,
		       int step) const;


  // The Drift Velocity (cm/ns)
  static float vDrift;
  // // The Drift Velocity (cm/ns) for MB1 Wheel1 (non fluxed chamber) 21-Dec-2006 SL
  // static float vDriftMB1W1;

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


