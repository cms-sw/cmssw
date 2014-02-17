#ifndef RecoLocalMuon_DTTTrigSyncFromDB_H
#define RecoLocalMuon_DTTTrigSyncFromDB_H

/** \class DTTTrigSyncFromDB
 *  Concrete implementation of a DTTTrigBaseSync.
 *  This class define the offset for RecHit building
 *  of data and simulation.
 *  The offset is computes as: 
 *  <br>
 *  offset = t0 + tTrig + wirePropCorr - tofCorr 
 *  <br>
 *  where: <br>
 *     - t0 from test pulses (taken from DB, it is assumed to be in ns; can be switched off)
 *     - ttrig from the fit of time boxrising edge (taken from DB, it is assumed to be in ns)
 *       (At the moment a single value is read for ttrig offset 
 *       but this may change in the future)
 *     - signal propagation along the wire (can be switched off):
 *       it is assumed the ttrig accounts on average for
 *       correction from the center of the wire to the frontend.
 *       Here we just have to correct for the distance of the hit from the wire center.
 *     - TOF correction (can be switched off for cosmics):
 *       the ttrig already accounts for average TOF correction, 
 *       depending on the granularity used for the ttrig computation we just have to correct for the
 *       TOF from the center of the chamber, SL, layer or wire to the hit position.
 *       NOTE: particles are assumed as coming from the IP.
 *
 *  The emulatorOffset is computed as:
 *  <br>
 *  offset = int(ttrig/BXspace)*BXspace + t0
 *  <br>
 *  where: <br>
 *     - t0 from test pulses (taken from DB, it is assumed to be in ns; can be switched off)
 *     - ttrig from the fit of time box rising edge (taken from DB, it is assumed to be in ns)
 *     - BXspace BX spacing (in ns). Can be configured.
 *   
 *  NOTE: this should approximate what is seen online by the BTI
 *
 *  $Date: 2009/12/07 17:22:18 $
 *  $Revision: 1.5 $
 *  \author G. Cerminara - INFN Torino
 */

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"


class DTLayer;
class DTWireId;
class DTT0;
class DTTtrig;


namespace edm {
  class ParameterSet;
}

class DTTTrigSyncFromDB : public DTTTrigBaseSync {
public:
  /// Constructor
  DTTTrigSyncFromDB(const edm::ParameterSet& config);

  /// Destructor
  virtual ~DTTTrigSyncFromDB();

  // Operations

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);


  /// Time (ns) to be subtracted to the digi time,
  /// Parameters are the layer and the wireId to which the
  /// digi is referred and the estimation of
  /// the 3D hit position (globPos)
  virtual double offset(const DTLayer* layer,
			const DTWireId& wireId,
			const GlobalPoint& globPos,
			double& tTrig,
			double& wirePropCorr,
			double& tofCorr);

  /// Time (ns) to be subtracted to the digi time.
  /// It does not take into account TOF and signal propagation along the wire
  double offset(const DTWireId& wireId);


  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// It does not take into account TOF and signal propagation along the wire
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - t0cell is the t0 from pulses
  virtual double emulatorOffset(const DTWireId& wireId,
				double &tTrig,
				double &t0cell);


 private:
  
  const DTT0 *tZeroMap;
  const DTTtrig *tTrigMap;
  // Set the verbosity level
  static bool debug;
  // The velocity of signal propagation along the wire (cm/ns)
  double theVPropWire;
  // Switch on/off the T0 correction from pulses
  bool doT0Correction;
  // Switch on/off the TOF correction for particles from IP
  bool doTOFCorrection;
  int theTOFCorrType;
  // Switch on/off the correction for the signal propagation along the wire
  bool doWirePropCorrection;
  int theWirePropCorrType;
  // spacing of BX in ns
  double theBXspace;

  std::string thetTrigLabel;

};
#endif

