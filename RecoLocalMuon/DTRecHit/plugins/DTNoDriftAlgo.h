#ifndef RecoLocalMuon_DTNoDriftAlgo_H
#define RecoLocalMuon_DTNoDriftAlgo_H

/** \class DTNoDriftAlgo
 *  Concrete implementation of DTRecHitBaseAlgo.
 *  Create pair of RecHits at fixed distance from
 *  the wire.
 *
 *  \author Martijn Mulders - CERN (martijn.mulders@cern.ch)
 *  based on DTLinearDriftAlgo
 */

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"



class DTNoDriftAlgo : public DTRecHitBaseAlgo {
 public:
  /// Constructor
  DTNoDriftAlgo(const edm::ParameterSet& config);

  /// Destructor
  ~DTNoDriftAlgo() override;

  // Operations

  /// Pass the Event Setup to the algo at each event
  void setES(const edm::EventSetup& setup) override;


  /// MM: Override virtual function from DTRecHitBaseAlgo--> for the NoDrift
  /// algorithm only a maximum of one hit per wire is allowed! 
  /// Build all hits in the range associated to the layerId, at the 1st step.
  edm::OwnVector<DTRecHit1DPair> reconstruct(const DTLayer* layer,
						     const DTLayerId& layerId,
						     const DTDigiCollection::Range& digiRange) override;


  /// First step in computation of Left/Right hits from a Digi.  
  /// The results are the local position (in DTLayer frame) of the
  /// Left and Right hit, and the error (which is common). Returns
  /// false on failure. The hit is assumed to be at the wire center.
  bool compute(const DTLayer* layer,
                       const DTDigi& digi,
                       LocalPoint& leftPoint,
                       LocalPoint& rightPoint,
                       LocalError& error) const override;


  /// Second step in hit position computation.
  /// It is the same as first step since the angular information is not used
  /// NOTE: Only position and error of the new hit are modified
  bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       DTRecHit1D& newHit1D) const override;


  /// Third (and final) step in hits position computation.
  /// Also the hit position along the wire is available
  /// and can be used to correct the drift time for particle
  /// TOF and propagation of signal along the wire. 
  /// NOTE: Only position and error of the new hit are modified
  bool compute(const DTLayer* layer,
                       const DTRecHit1D& recHit1D,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       DTRecHit1D& newHit1D) const override;


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
  const float fixedDrift;

  // The resolution on the Hits (cm)
  const float hitResolution;

  // Times below MinTime (ns) are considered as coming from previous BXs.
  const float minTime;

  // Times above MaxTime (ns) are considered as coming from following BXs
  const float maxTime;

  // Switch on/off the verbosity
  const bool debug;
};
#endif


