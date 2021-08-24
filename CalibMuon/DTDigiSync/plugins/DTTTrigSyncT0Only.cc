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
                double& tofCorr) const override;

  double offset(const DTWireId& wireId) const override;

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// Returns just 0 in this implementation of the plugin
  double emulatorOffset(const DTWireId& wireId, double& tTrig, double& t0cell) const override;

private:
  const DTT0* tZeroMap;

  // Set the verbosity level
  const bool debug;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

#include <iostream>

using namespace std;
using namespace edm;

DTTTrigSyncT0Only::DTTTrigSyncT0Only(const ParameterSet& config) : debug(config.getUntrackedParameter<bool>("debug")) {}

DTTTrigSyncT0Only::~DTTTrigSyncT0Only() {}

void DTTTrigSyncT0Only::setES(const EventSetup& setup) {
  ESHandle<DTT0> t0;
  setup.get<DTT0Rcd>().get(t0);
  tZeroMap = &*t0;

  if (debug) {
    cout << "[DTTTrigSyncT0Only] T0 version: " << t0->version() << endl;
  }
}

double DTTTrigSyncT0Only::offset(const DTLayer* layer,
                                 const DTWireId& wireId,
                                 const GlobalPoint& globPos,
                                 double& tTrig,
                                 double& wirePropCorr,
                                 double& tofCorr) const {
  tTrig = offset(wireId);
  wirePropCorr = 0;
  tofCorr = 0;

  if (debug) {
    cout << "[DTTTrigSyncT0Only] Offset (ns): " << tTrig + wirePropCorr - tofCorr << endl
         << "      various contributions are: "
         << endl
         //<< "      tZero (ns):   " << t0 << endl
         << "      Propagation along wire delay (ns): " << wirePropCorr << endl
         << "      TOF correction (ns): " << tofCorr << endl
         << endl;
  }
  //The global offset is the sum of various contributions
  return tTrig + wirePropCorr - tofCorr;
}

double DTTTrigSyncT0Only::offset(const DTWireId& wireId) const {
  float t0 = 0;
  float t0rms = 0;
  tZeroMap->get(wireId, t0, t0rms, DTTimeUnits::ns);

  return t0;
}

double DTTTrigSyncT0Only::emulatorOffset(const DTWireId& wireId, double& tTrig, double& t0cell) const {
  tTrig = 0.;
  t0cell = 0.;
  return 0.;
}

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

DEFINE_EDM_PLUGIN(DTTTrigSyncFactory, DTTTrigSyncT0Only, "DTTTrigSyncT0Only");
