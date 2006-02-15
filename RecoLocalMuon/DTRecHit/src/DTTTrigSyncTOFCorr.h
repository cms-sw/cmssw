#ifndef RecoLocalMuon_DTTTrigSyncTOFCorr_H
#define RecoLocalMuon_DTTTrigSyncTOFCorr_H

/** \class DTTTrigSyncTOFCorr
 *  Concrete implementation of a DTTTrigBaseSync.
 *  This class define the offsets for RecHit building
 *  coherently to the digitization realized with the
 *  DTDigiSyncTOFCorr module.
 *  The offset is computes as:
 *  offset = tTrig + wirePropCorr - tofCorr
 *  where:
 *      _ tTrig is a fixed offset defined in tTrig parameter
 *        (default 500 ns)
 *      _ wirePropCorr is the correction for the signal propagation along the wire
 *      _ tofCorr is the correction for the TOF of the particle set according to
 *        tofCorrType parameter:
 *        0: tofCorrType = TOF from IP to 3D Hit position (globPos)
 *        1: tofCorrType = TOF correction for distance difference
 *                         between 3D center of the chamber and hit position
 *        2: tofCorrType = TOF correction for distance difference
 *                         between 3D center of the wire and hit position
 *                         (This mode in available for backward compatibility)
 *
 *
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTTTrigBaseSync.h"



class DTLayer;
class DTWireId;

namespace edm {
  class ParameterSet;
}

class DTTTrigSyncTOFCorr : public DTTTrigBaseSync {
public:
  /// Constructor
  DTTTrigSyncTOFCorr(const edm::ParameterSet& config);

  /// Destructor
  virtual ~DTTTrigSyncTOFCorr();

  // Operations

  /// Time (ns) to be subtracted to the digi time,
  /// Parameters are the layer and the wireId to which the
  /// digi is referred and the estimation of
  /// the 3D hit position (globPos)
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - wirePropCorr is the delay for signal propagation along the wire
  ///     - tofCorr is the correction due to the particle TOF 
  virtual double offset(const DTLayer* layer,
			const DTWireId& wireId,
			const GlobalPoint& globPos,
			double& tTrig,
			double& wirePropCorr,
			double& tofCorr);

 private:
  // The fixed t_trig to be subtracted to digi time (ns)
  static double theTTrig;
  // Velocity of signal propagation along the wire (cm/ns)
  // For the value
  // cfr. CMS-IN 2000-021:   (2.56+-0.17)x1e8 m/s
  //      CMS NOTE 2003-17:  (0.244)  m/ns = 24.4 cm/ns
  static double theVPropWire;
  
  // Select the mode for TOF correction:
  //     0: tofCorr = TOF from IP to 3D Hit position (globPos)
  //     1: tofCorr = TOF correction for distance difference
  //                  between 3D center of the chamber and hit position
  //     2: tofCorr = TOF correction for distance difference
  //                  between 3D center of the wire and hit position
  static int theTOFCorrType;

  // Set the verbosity level
  static bool debug;
};
#endif

