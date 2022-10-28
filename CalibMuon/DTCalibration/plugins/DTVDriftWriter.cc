
/*
 *  See header file for a description of this class.
 *
 *  Author of original version: M. Giunta
 *  \author A. Vilela Pereira
 */

#include "DTVDriftWriter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DTObjects/interface/DTRecoConditions.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"
#include "CalibMuon/DTCalibration/interface/DTVDriftPluginFactory.h"
#include "CalibMuon/DTCalibration/interface/DTVDriftBaseAlgo.h"

#include <string>
#include <vector>

using namespace std;
using namespace edm;

DTVDriftWriter::DTVDriftWriter(const ParameterSet& pset)
    : mTimeMapToken_(esConsumes<edm::Transition::BeginRun>()),
      vDriftMapToken_(esConsumes<edm::Transition::BeginRun>()),
      dtGeomToken_(esConsumes<edm::Transition::BeginRun>()),
      granularity_(pset.getUntrackedParameter<string>("calibGranularity", "bySL")),
      mTimeMap_(nullptr),
      vDriftMap_(nullptr),
      vDriftAlgo_{DTVDriftPluginFactory::get()->create(pset.getParameter<string>("vDriftAlgo"),
                                                       pset.getParameter<ParameterSet>("vDriftAlgoConfig"),
                                                       consumesCollector())} {
  LogVerbatim("Calibration") << "[DTVDriftWriter]Constructor called!";

  if (granularity_ != "bySL")
    throw cms::Exception("Configuration")
        << "[DTVDriftWriter] Check parameter calibGranularity: " << granularity_ << " option not available.";

  readLegacyVDriftDB = pset.getParameter<bool>("readLegacyVDriftDB");
  writeLegacyVDriftDB = pset.getParameter<bool>("writeLegacyVDriftDB");
}

DTVDriftWriter::~DTVDriftWriter() { LogVerbatim("Calibration") << "[DTVDriftWriter]Destructor called!"; }

void DTVDriftWriter::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  // Get the map of vdrift from the Setup
  if (readLegacyVDriftDB) {
    mTimeMap_ = &setup.getData(mTimeMapToken_);
  } else {
    vDriftMap_ = &setup.getData(vDriftMapToken_);
    // Consistency check: no parametrization is implemented for the time being
    int version = vDriftMap_->version();
    if (version != 1) {
      throw cms::Exception("Configuration") << "only version 1 is presently supported for VDriftDB";
    }
  }

  // Get geometry from Event Setup
  dtGeom_ = setup.getHandle(dtGeomToken_);

  // Pass EventSetup to concrete implementation
  vDriftAlgo_->setES(setup);
}

void DTVDriftWriter::endJob() {
  // Create the object to be written to DB
  std::unique_ptr<DTMtime> mTimeNewMap;
  std::unique_ptr<DTRecoConditions> vDriftNewMap;
  if (writeLegacyVDriftDB) {
    mTimeNewMap = std::make_unique<DTMtime>();
  } else {
    vDriftNewMap = std::make_unique<DTRecoConditions>();
    vDriftNewMap->setFormulaExpr("[0]");
    //vDriftNewMap->setFormulaExpr("[0]*(1-[1]*x)"); // add parametrization for dependency along Y
    vDriftNewMap->setVersion(1);
  }

  if (granularity_ == "bySL") {
    // Get all the sls from the geometry
    const vector<const DTSuperLayer*>& superLayers = dtGeom_->superLayers();
    auto sl = superLayers.begin();
    auto sl_end = superLayers.end();
    for (; sl != sl_end; ++sl) {
      DTSuperLayerId slId = (*sl)->id();

      // Compute vDrift
      float vDriftNew = -1.;
      float resolutionNew = -1;
      try {
        dtCalibration::DTVDriftData vDriftData = vDriftAlgo_->compute(slId);
        vDriftNew = vDriftData.vdrift;
        resolutionNew = vDriftData.resolution;
        LogVerbatim("Calibration") << "vDrift for: " << slId << " Mean " << vDriftNew << " Resolution "
                                   << resolutionNew;
      } catch (cms::Exception& e) {  // Failure to compute new value, fall back to old table
        LogError("Calibration") << e.explainSelf();
        if (readLegacyVDriftDB) {  //...reading old db format...
          int status = mTimeMap_->get(slId, vDriftNew, resolutionNew, DTVelocityUnits::cm_per_ns);
          if (status == 0) {  // not found; silently skip this SL
            continue;
          }
        } else {  //...reading new db format
          try {
            vDriftNew = vDriftMap_->get(DTWireId(slId.rawId()));
          } catch (cms::Exception& e2) {
            // not found; silently skip this SL
            continue;
          }
        }
        LogVerbatim("Calibration") << "Keep original vDrift for: " << slId << " Mean " << vDriftNew << " Resolution "
                                   << resolutionNew;
      }

      // Add value to the vdrift table
      if (writeLegacyVDriftDB) {
        mTimeNewMap->set(slId, vDriftNew, resolutionNew, DTVelocityUnits::cm_per_ns);
      } else {
        vector<double> params = {vDriftNew};
        vDriftNewMap->set(DTWireId(slId.rawId()), params);
      }
    }  // End of loop on superlayers
  }

  // Write the vDrift object to DB
  LogVerbatim("Calibration") << "[DTVDriftWriter]Writing vdrift object to DB!";
  if (writeLegacyVDriftDB) {
    string record = "DTMtimeRcd";
    DTCalibDBUtils::writeToDB<DTMtime>(record, *mTimeNewMap);
  } else {
    DTCalibDBUtils::writeToDB<DTRecoConditions>("DTRecoConditionsVdriftRcd", *vDriftNewMap);
  }
}
