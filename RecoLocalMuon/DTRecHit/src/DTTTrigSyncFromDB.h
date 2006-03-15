#ifndef RecoLocalMuon_DTTTrigSyncFromDB_H
#define RecoLocalMuon_DTTTrigSyncFromDB_H

/** \class DTTTrigSyncFromDB
 *  Concrete implementation of a DTTTrigBaseSync.
 *  This plugin reads only the t0 from pulses from the DB.
 *
 *
 *  $Date: 2006/03/14 13:02:42 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTTTrigBaseSync.h"



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

 private:
  
  const DTT0 *tZeroMap;
  const DTTtrig *tTrigMap;
  // Set the verbosity level
  static bool debug;
};
#endif

