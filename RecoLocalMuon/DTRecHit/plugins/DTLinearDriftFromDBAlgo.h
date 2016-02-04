#ifndef RecoLocalMuon_DTLinearDrifFromDBtAlgo_H
#define RecoLocalMuon_DTLinearDriftFromDBAlgo_H

/** \class DTLinearDriftFromDBAlgo
 *  Concrete implementation of DTRecHitBaseAlgo.
 *  Compute drift distance using constant drift velocity
 *  read from database.
 *
 *  $Date: 2011/02/22 16:23:00 $
 *  $Revision: 1.3 $
 *  \author S.Bolognesi - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"

class DTMtime;

class DTLinearDriftFromDBAlgo : public DTRecHitBaseAlgo {
 public:
  /// Constructor
  DTLinearDriftFromDBAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~DTLinearDriftFromDBAlgo();

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

  //Map of meantimes
   const DTMtime *mTimeMap;
 
  // Times below MinTime (ns) are considered as coming from previous BXs.
  static float minTime;

  // Times above MaxTime (ns) are considered as coming from following BXs
  static float maxTime;
  
  // Perform a correction to vDrift for the external wheels
  bool doVdriftCorr;

  // Switch recalculating hit parameters from digi time in Step 2 
  // (when off, Step 2 does nothing)
  bool stepTwoFromDigi;

  // Switch on/off the verbosity
  static bool debug;
};
#endif


