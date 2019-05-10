#ifndef RecoLocalMuon_DTTTrigBaseSync_H
#define RecoLocalMuon_DTTTrigBaseSync_H

/** \class DTTTrigBaseSync
 *  Base class to define the offsets for 1D DT RecHit building
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class DTLayer;
class DTWireId;

namespace edm {
  class EventSetup;
}

class DTTTrigBaseSync {
public:
  /// Constructor
  DTTTrigBaseSync();

  /// Destructor
  virtual ~DTTTrigBaseSync();

  // Operations

  /// Pass the Event Setup to the synchronization module at each event
  virtual void setES(const edm::EventSetup& setup) = 0;

  /// Time (ns) to be subtracted to the digi time.
  /// Parameters are the layer and the wireId to which the
  /// digi is referred and the estimation of
  /// the 3D hit position (globPos)
  double offset(const DTLayer* layer, const DTWireId& wireId, const GlobalPoint& globalPos) const;

  /// Time (ns) to be subtracted to the digi time.
  /// It does not take into account TOF and signal propagation along the wire
  virtual double offset(const DTWireId& wireId) const = 0;

  /// Time to be subtracted to the digi time,
  /// Parameters are the layer and the wireId to which the
  /// digi is referred and the estimation of
  /// the 3D hit position (globPos)
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - wirePropCorr is the delay for signal propagation along the wire
  ///     - tofCorr is the correction due to the particle TOF
  virtual double offset(const DTLayer* layer,
                        const DTWireId& wireId,
                        const GlobalPoint& globalPos,
                        double& tTrig,
                        double& wirePropCorr,
                        double& tofCorr) const = 0;

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// It does not take into account TOF and signal propagation along the wire
  virtual double emulatorOffset(const DTWireId& wireId) const;

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// It does not take into account TOF and signal propagation along the wire
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - t0cell is the t0 from pulses
  virtual double emulatorOffset(const DTWireId& wireId, double& tTrig, double& t0cell) const = 0;
};
#endif
