#ifndef RecoLocalMuon_DTTTrigSyncT0Only_H
#define RecoLocalMuon_DTTTrigSyncT0Only_H

/** \class DTTTrigSyncT0Only
 *  Concrete implementation of a DTTTrigBaseSync.
 *  This plugin reads only the t0 from pulses from the DB.
 *
 *
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
  ~DTTTrigSyncT0Only() override;

  // Operations

  /// Pass the Event Setup to the algo at each event
  void setES(const edm::EventSetup& setup) override;


  /// Time (ns) to be subtracted to the digi time,
  /// Parameters are the layer and the wireId to which the
  /// digi is referred and the estimation of
  /// the 3D hit position (globPos)
  double offset(const DTLayer* layer,
			const DTWireId& wireId,
			const GlobalPoint& globPos,
			double& tTrig,
			double& wirePropCorr,
			double& tofCorr) override;

  double offset(const DTWireId& wireId) override;

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// Returns just 0 in this implementation of the plugin
  double emulatorOffset(const DTWireId& wireId,
				double &tTrig,
				double &t0cell) override;


 private:
  
  const DTT0 *tZeroMap;

  // Set the verbosity level
  const bool debug;
};
#endif

