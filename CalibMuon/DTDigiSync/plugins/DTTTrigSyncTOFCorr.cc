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
  ~DTTTrigSyncTOFCorr() override;

  // Operations

  /// Pass the Event Setup to the algo at each event
  void setES(const edm::EventSetup& setup) override {}

  /// Time (ns) to be subtracted to the digi time,
  /// Parameters are the layer and the wireId to which the
  /// digi is referred and the estimation of
  /// the 3D hit position (globPos)
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - wirePropCorr is the delay for signal propagation along the wire
  ///     - tofCorr is the correction due to the particle TOF
  double offset(const DTLayer* layer,
                const DTWireId& wireId,
                const GlobalPoint& globPos,
                double& tTrig,
                double& wirePropCorr,
                double& tofCorr) const override;

  double offset(const DTWireId& wireId) const override;

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// It does not take into account TOF and signal propagation along the wire
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - t0cell is the t0 from pulses (always 0 in this case)
  double emulatorOffset(const DTWireId& wireId, double& tTrig, double& t0cell) const override;

private:
  // The fixed t_trig to be subtracted to digi time (ns)
  const double theTTrig;
  // Velocity of signal propagation along the wire (cm/ns)
  // For the value
  // cfr. CMS-IN 2000-021:   (2.56+-0.17)x1e8 m/s
  //      CMS NOTE 2003-17:  (0.244)  m/ns = 24.4 cm/ns
  const double theVPropWire;

  // Select the mode for TOF correction:
  //     0: tofCorr = TOF from IP to 3D Hit position (globPos)
  //     1: tofCorr = TOF correction for distance difference
  //                  between 3D center of the chamber and hit position
  //     2: tofCorr = TOF correction for distance difference
  //                  between 3D center of the wire and hit position
  const int theTOFCorrType;

  // Set the verbosity level
  const bool debug;
  // spacing of BX in ns
  double theBXspace;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include <iostream>

using namespace std;

DTTTrigSyncTOFCorr::DTTTrigSyncTOFCorr(const edm::ParameterSet& config)
    :                                                  // The fixed t0 (or t_trig) to be subtracted to digi time (ns)
      theTTrig(config.getParameter<double>("tTrig")),  // FIXME: Default was 500 ns
      // Velocity of signal propagation along the wire (cm/ns)
      theVPropWire(config.getParameter<double>("vPropWire")),  // FIXME: Default was 24.4 cm/ns
      // Select the mode for TOF correction:
      //     0: tofCorr = TOF from IP to 3D Hit position (globPos)
      //     1: tofCorr = TOF correction for distance difference
      //                  between 3D center of the chamber and hit position
      //                  (default)
      //     2: tofCorr = TOF correction for distance difference
      //                  between 3D center of the wire and hit position
      //                  (This mode in available for backward compatibility)
      theTOFCorrType(config.getParameter<int>("tofCorrType")),  // FIXME: Default was 1
      debug(config.getUntrackedParameter<bool>("debug")),
      theBXspace(config.getUntrackedParameter<double>("bxSpace", 25.))  // spacing of BX in ns
{}

DTTTrigSyncTOFCorr::~DTTTrigSyncTOFCorr() {}

double DTTTrigSyncTOFCorr::offset(const DTLayer* layer,
                                  const DTWireId& wireId,
                                  const GlobalPoint& globPos,
                                  double& tTrig,
                                  double& wirePropCorr,
                                  double& tofCorr) const {
  tTrig = offset(wireId);

  //Compute the time spent in signal propagation along wire.
  // NOTE: the FE is always at y>0
  float halfL = layer->specificTopology().cellLenght() / 2;
  float wireCoord = layer->toLocal(globPos).y();
  float propgL = halfL - wireCoord;
  wirePropCorr = propgL / theVPropWire;

  // Compute TOF correction treating it accordingly to
  // the tofCorrType card
  float flightToHit = globPos.mag();
  static const float cSpeed = 29.9792458;  // cm/ns
  tofCorr = 0.;
  switch (theTOFCorrType) {
    case 0: {
      // In this mode the subtraction of the TOF from IP to
      // estimate 3D hit digi position is done here
      // (No correction is needed anymore)
      tofCorr = -flightToHit / cSpeed;
      break;
    }
    case 1: {
      // Correction for TOF from the center of the chamber to hit position
      const DTChamber* chamber = layer->chamber();
      double flightToChamber = chamber->surface().position().mag();
      tofCorr = (flightToChamber - flightToHit) / cSpeed;
      break;
    }
    case 2: {
      // TOF from 3D center of the wire to hit position
      float flightToWire =
          layer->toGlobal(LocalPoint(layer->specificTopology().wirePosition(wireId.wire()), 0., 0.)).mag();
      tofCorr = (flightToWire - flightToHit) / cSpeed;
      break;
    }
    default: {
      throw cms::Exception("[DTTTrigSyncTOFCorr]")
          << " Invalid parameter: tofCorrType = " << theTOFCorrType << std::endl;
      break;
    }
  }

  if (debug) {
    cout << "[DTTTrigSyncTOFCorr] Offset (ns): " << tTrig + wirePropCorr - tofCorr << endl
         << "      various contributions are: " << endl
         << "      tTrig (ns):   " << tTrig << endl
         << "      Propagation along wire delay (ns): " << wirePropCorr << endl
         << "      TOF correction (ns): " << tofCorr << endl
         << endl;
  }
  //The global offset is the sum of various contributions
  return tTrig + wirePropCorr - tofCorr;
}

double DTTTrigSyncTOFCorr::offset(const DTWireId& wireId) const {
  // Correction for the float to int conversion
  // (half a bin on average) in DTDigi constructor
  static const float f2i_convCorr = (25. / 64.);  // ns
  //FIXME: This should be considered only if the Digi is constructed from a float....

  // The tTrig is taken from a parameter
  return theTTrig - f2i_convCorr;
}

double DTTTrigSyncTOFCorr::emulatorOffset(const DTWireId& wireId, double& tTrig, double& t0cell) const {
  tTrig = theTTrig;
  t0cell = 0.;

  return int(tTrig / theBXspace) * theBXspace;
}

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

DEFINE_EDM_PLUGIN(DTTTrigSyncFactory, DTTTrigSyncTOFCorr, "DTTTrigSyncTOFCorr");
