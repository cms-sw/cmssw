#ifndef RecoLocalMuon_DTTTrigSyncT0Only_H
#define RecoLocalMuon_DTTTrigSyncT0Only_H

/** \class DTTTrigSyncT0Only
 *  Concrete implementation of a DTTTrigBaseSync.
 *  This plugin reads only the t0 from pulses from the DB.
 *
 *
 *  $Date: 2009/10/21 17:05:47 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"



class DTLayer;
class DTWireId;
class DTT0;

namespace edm {
  class ParameterSet;
}

class DTTTrigSyncT0Only : public DTTTrigBaseSync {
public:
  /// Constructor
  DTTTrigSyncT0Only(const edm::ParameterSet& config);

  /// Destructor
  virtual ~DTTTrigSyncT0Only();

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

  virtual double offset(const DTWireId& wireId);

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// Returns just 0 in this implementation of the plugin
  virtual double emulatorOffset(const DTWireId& wireId,
				double &tTrig,
				double &t0cell);


 private:
  
  const DTT0 *tZeroMap;

  // Set the verbosity level
  static bool debug;
};
#endif

