#ifndef RecoLocalMuon_DTTTrigSyncTOFCorr_H
#define RecoLocalMuon_DTTTrigSyncTOFCorr_H

/** \class DTTTrigSyncTOFCorr
 *  Concrete implementation of a DTTTrigBaseSync.
 *  This class define the offsets for RecHit building
 *  coherently to the digitization realized with the
 *  DTDigiSyncTOFCorr module.
 *  The offset is computes as:<br>
 *  offset = tTrig + wirePropCorr - tofCorr<br>
 *  where:<br>
 *      - tTrig is a fixed offset defined in tTrig parameter
 *        (default 500 ns)<br>
 *      - wirePropCorr is the correction for the signal propagation along the wire<br>
 *      - tofCorr is the correction for the TOF of the particle set according to
 *        tofCorrType parameter:<br>
 *        0: tofCorrType = TOF from IP to 3D Hit position (globPos)<br>
 *        1: tofCorrType = TOF correction for distance difference
 *                         between 3D center of the chamber and hit position<br>
 *        2: tofCorrType = TOF correction for distance difference
 *                         between 3D center of the wire and hit position
 *                         (This mode in available for backward compatibility)<br>
 *
 *  The emulatorOffset is computed as:
 *  <br>
 *  offset = int(ttrig/BXspace)*BXspace
 *  <br>
 *  where: <br>
 *     - ttrig from the fit of time box rising edge (taken from configuration, it is assumed to be in ns)
 *     - BXspace BX spacing (in ns). Taken from configuration (default 25ns).
 *   
 *  NOTE: this should approximate what is seen online by the BTI
 *
 *
 *
 *  $Date: 2009/10/21 17:05:47 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"



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

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup) {}


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

  virtual double offset(const DTWireId& wireId);

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// It does not take into account TOF and signal propagation along the wire
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - t0cell is the t0 from pulses (always 0 in this case)
  virtual double emulatorOffset(const DTWireId& wireId,
				double &tTrig,
				double &t0cell);

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
  // spacing of BX in ns
  double theBXspace;
};
#endif

