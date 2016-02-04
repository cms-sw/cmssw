#ifndef RecoLocalMuon_DTParametrizedDriftAlgo_H
#define RecoLocalMuon_DTParametrizedDriftAlgo_H

/** \class DTParametrizedDriftAlgo
 *  Concrete implementation of DTRecHitBaseAlgo.
 *  Compute drift distance using the CIEMAT (by P.Garcia Abia and J. Puerta)
 *  parametrization of the cell behavior obtained with GARFIELD
 *
 *  $Date: 2007/04/19 11:08:17 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"

class MagneticField;

class DTParametrizedDriftAlgo : public DTRecHitBaseAlgo {
 public:
  /// Constructor
  DTParametrizedDriftAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~DTParametrizedDriftAlgo();

  // Operations

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);

    
  /// First step in computation of Left/Right hits from a Digi.  
  /// The results are the local position (in DTLayer frame) of the
  /// Left and Right hit, and the error (which is common). 
  /// The center of the wire is assumed as hit coordinate along y.
  /// Returns false on failure. 
  virtual bool compute(const DTLayer* layer,
                       const DTDigi& digi,
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const;


  /// Second step.
  /// The impact angle is given as input, and it's used to improve the hit
  /// position (and relative error). The angle is defined in radians, with
  /// respect to the perpendicular to the layer plane. Given the local direction,
  /// angle=atan(dir.x()/-dir.z()) . This can be used when a SL segment is
  /// built, so the impact angle is known but the position along wire is not.
  /// NOTE: Only position and error of the new hit are modified
  virtual bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
		       DTRecHit1D& newHit1D) const;


  /// Third (and final) step in hits position computation.
  /// In addition the the angle, also the global position of the hit is given
  /// as input. This allows to get the magnetic field at the hit position (and
  /// not only that at the center of the wire). Also the position along the
  /// wire is available and can be used to correct the drift time for particle
  /// TOF and propagation of signal along the wire.
  /// NOTE: Only position and error of the new hit are modified
  virtual bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       DTRecHit1D& newHit1D) const;


 private:
  // Interpolate parametrization function
  static bool interpolate;

  // Times below MinTime (ns) are considered as coming from previous BXs.
  static float minTime;
  
  // Times above MaxTime (ns) are considered as coming from following BXs
  static float maxTime;
  
  // Do the actual work.
  virtual bool compute(const DTLayer* layer,
		       const DTWireId& wireId,
		       const float digiTime,
		       const float& angle,
		       const GlobalPoint& globPos, 
		       LocalPoint& leftPoint,
		       LocalPoint& rightPoint,
		       LocalError& error,
		       int step) const;

  // Interface to the method which does the actual work suited for 2nd and 3rd steps 
  virtual bool compute(const DTLayer* layer,
		       const DTWireId& wireId,
		       const float digiTime,
		       const float& angle,
		       const GlobalPoint& globPos, 
		       DTRecHit1D& newHit1D,
		       int step) const;

  // Switch on/off the verbosity
  static bool debug;


  // Pointer to the magnetic field (read from ES once per event)
  const MagneticField * magField;


};
#endif


