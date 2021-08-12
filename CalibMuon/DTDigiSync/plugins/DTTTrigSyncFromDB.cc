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
  ~DTTTrigSyncFromDB() override;

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

  /// Time (ns) to be subtracted to the digi time.
  /// It does not take into account TOF and signal propagation along the wire
  double offset(const DTWireId& wireId) const override;

  /// Time (ns) to be subtracted to the digi time for emulation purposes
  /// It does not take into account TOF and signal propagation along the wire
  /// It also returns the different contributions separately:
  ///     - tTrig is the offset (t_trig)
  ///     - t0cell is the t0 from pulses
  double emulatorOffset(const DTWireId& wireId, double& tTrig, double& t0cell) const override;

private:
  const DTT0* tZeroMap;
  const DTTtrig* tTrigMap;
  // Set the verbosity level
  const bool debug;
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
  std::string thet0Label;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <iostream>

using namespace std;
using namespace edm;

DTTTrigSyncFromDB::DTTTrigSyncFromDB(const ParameterSet& config)
    : debug(config.getUntrackedParameter<bool>("debug")),
      // The velocity of signal propagation along the wire (cm/ns)
      theVPropWire(config.getParameter<double>("vPropWire")),
      // Switch on/off the T0 correction from pulses
      doT0Correction(config.getParameter<bool>("doT0Correction")),
      // Switch on/off the TOF correction for particles from IP
      doTOFCorrection(config.getParameter<bool>("doTOFCorrection")),
      theTOFCorrType(config.getParameter<int>("tofCorrType")),
      // Switch on/off the correction for the signal propagation along the wire
      doWirePropCorrection(config.getParameter<bool>("doWirePropCorrection")),
      theWirePropCorrType(config.getParameter<int>("wirePropCorrType")),
      // spacing of BX in ns
      theBXspace(config.getUntrackedParameter<double>("bxSpace", 25.)),
      thetTrigLabel(config.getParameter<string>("tTrigLabel")),
      thet0Label(config.getParameter<string>("t0Label")) {}

DTTTrigSyncFromDB::~DTTTrigSyncFromDB() {}

void DTTTrigSyncFromDB::setES(const EventSetup& setup) {
  if (doT0Correction) {
    // Get the map of t0 from pulses from the Setup
    ESHandle<DTT0> t0Handle;
    setup.get<DTT0Rcd>().get(thet0Label, t0Handle);
    tZeroMap = &*t0Handle;
    if (debug) {
      cout << "[DTTTrigSyncFromDB] t0 version: " << tZeroMap->version() << endl;
    }
  }

  // Get the map of ttrig from the Setup
  ESHandle<DTTtrig> ttrigHandle;
  setup.get<DTTtrigRcd>().get(thetTrigLabel, ttrigHandle);
  tTrigMap = &*ttrigHandle;
  if (debug) {
    cout << "[DTTTrigSyncFromDB] ttrig version: " << tTrigMap->version() << endl;
  }
}

double DTTTrigSyncFromDB::offset(const DTLayer* layer,
                                 const DTWireId& wireId,
                                 const GlobalPoint& globPos,
                                 double& tTrig,
                                 double& wirePropCorr,
                                 double& tofCorr) const {
  // Correction for the float to int conversion while writeing the ttrig in ns into an int variable
  // (half a bin on average)
  // FIXME: this should disappear as soon as the ttrig object will become a float
  //   static const float f2i_convCorr = (25./64.); // ns //FIXME: check how the conversion is performed

  tTrig = offset(wireId);

  // Compute the time spent in signal propagation along wire.
  // NOTE: the FE is always at y>0
  wirePropCorr = 0;
  if (doWirePropCorrection) {
    switch (theWirePropCorrType) {
        // The ttrig computed from the timebox accounts on average for the signal propagation time
        // from the center of the wire to the frontend. Here we just have to correct for
        // the distance of the hit from the wire center.
      case 0: {
        float wireCoord = layer->toLocal(globPos).y();
        wirePropCorr = -wireCoord / theVPropWire;
        break;
        // FIXME: What if hits used for the time box are not distributed uniformly along the wire?
      }
      //On simulated data you need to subtract the total propagation time
      case 1: {
        float halfL = layer->specificTopology().cellLenght() / 2;
        float wireCoord = layer->toLocal(globPos).y();
        float propgL = halfL - wireCoord;
        wirePropCorr = propgL / theVPropWire;
        break;
      }
      default: {
        throw cms::Exception("[DTTTrigSyncFromDB]")
            << " Invalid parameter: wirePropCorrType = " << theWirePropCorrType << std::endl;
        break;
      }
    }
  }

  // Compute TOF correction:
  tofCorr = 0.;
  // TOF Correction can be switched off with appropriate parameter
  if (doTOFCorrection) {
    float flightToHit = globPos.mag();
    static const float cSpeed = 29.9792458;  // cm/ns
    switch (theTOFCorrType) {
      case 0: {
        // The ttrig computed from the real data accounts on average for the TOF correction
        // Depending on the granularity used for the ttrig computation we just have to correct for the
        // TOF from the center of the chamber, SL, layer or wire to the hit position.
        // At the moment only SL granularity is considered
        // Correction for TOF from the center of the SL to hit position
        const DTSuperLayer* sl = layer->superLayer();
        double flightToSL = sl->surface().position().mag();
        tofCorr = (flightToSL - flightToHit) / cSpeed;
        break;
      }
      case 1: {
        // On simulated data you need to consider only the TOF from 3D center of the wire to hit position
        // (because the TOF from the IP to the wire has been already subtracted in the digitization:
        // SimMuon/DTDigitizer/DTDigiSyncTOFCorr.cc corrType=2)
        float flightToWire =
            layer->toGlobal(LocalPoint(layer->specificTopology().wirePosition(wireId.wire()), 0., 0.)).mag();
        tofCorr = (flightToWire - flightToHit) / cSpeed;
        break;
      }
      default: {
        throw cms::Exception("[DTTTrigSyncFromDB]")
            << " Invalid parameter: tofCorrType = " << theTOFCorrType << std::endl;
        break;
      }
    }
  }

  if (debug) {
    cout << "[DTTTrigSyncFromDB] Channel: " << wireId << endl
         << "      Offset (ns): " << tTrig + wirePropCorr - tofCorr << endl
         << "      various contributions are: " << endl
         << "      tTrig + t0 (ns):   " << tTrig
         << endl
         //<< "      tZero (ns):   " << t0 << endl
         << "      Propagation along wire delay (ns): " << wirePropCorr << endl
         << "      TOF correction (ns): " << tofCorr << endl
         << endl;
  }
  //The global offset is the sum of various contributions
  return tTrig + wirePropCorr - tofCorr;
}

double DTTTrigSyncFromDB::offset(const DTWireId& wireId) const {
  float t0 = 0;
  float t0rms = 0;
  if (doT0Correction) {
    // Read the t0 from pulses for this wire (ns)
    tZeroMap->get(wireId, t0, t0rms, DTTimeUnits::ns);
  }

  // Read the ttrig for this wire
  float ttrigMean = 0;
  float ttrigSigma = 0;
  float kFactor = 0;
  // FIXME: should check the return value of the DTTtrigRcd::get(..) method
  if (tTrigMap->get(wireId.superlayerId(), ttrigMean, ttrigSigma, kFactor, DTTimeUnits::ns) != 0) {
    cout << "[DTTTrigSyncFromDB]*Error: ttrig not found for SL: " << wireId.superlayerId() << endl;
    //     FIXME: LogError.....
  }

  return t0 + ttrigMean + kFactor * ttrigSigma;
}

double DTTTrigSyncFromDB::emulatorOffset(const DTWireId& wireId, double& tTrig, double& t0cell) const {
  float t0 = 0;
  float t0rms = 0;
  if (doT0Correction) {
    // Read the t0 from pulses for this wire (ns)
    tZeroMap->get(wireId, t0, t0rms, DTTimeUnits::ns);
  }

  // Read the ttrig for this wire
  float ttrigMean = 0;
  float ttrigSigma = 0;
  float kFactor = 0;
  // FIXME: should check the return value of the DTTtrigRcd::get(..) method
  if (tTrigMap->get(wireId.superlayerId(), ttrigMean, ttrigSigma, kFactor, DTTimeUnits::ns) != 0) {
    cout << "[DTTTrigSyncFromDB]*Error: ttrig not found for SL: " << wireId.superlayerId() << endl;
    //     FIXME: LogError.....
  }

  tTrig = ttrigMean + kFactor * ttrigSigma;
  t0cell = t0;

  return int(tTrig / theBXspace) * theBXspace + t0cell;
}

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

DEFINE_EDM_PLUGIN(DTTTrigSyncFactory, DTTTrigSyncFromDB, "DTTTrigSyncFromDB");
