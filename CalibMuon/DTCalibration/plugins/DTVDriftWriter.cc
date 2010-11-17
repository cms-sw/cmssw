
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/09 22:44:10 $
 *  $Revision: 1.7 $
 *  Author of original version: M. Giunta
 *  \author A. Vilela Pereira
 */

#include "DTVDriftWriter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"

#include "CalibMuon/DTCalibration/interface/vDriftHistos.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"
#include "CalibMuon/DTCalibration/interface/DTVDriftPluginFactory.h"
#include "CalibMuon/DTCalibration/interface/DTVDriftBaseAlgo.h"

#include <string>
#include <vector>

using namespace std;
using namespace edm;

DTVDriftWriter::DTVDriftWriter(const ParameterSet& pset):
  granularity_( pset.getUntrackedParameter<string>("calibGranularity","bySL") ),
  vDriftDef_( pset.getUntrackedParameter<double>("vDrift",0.) ),
  vDriftResoDef_( pset.getUntrackedParameter<double>("vDriftResolution",0.) ) {

  LogVerbatim("Calibration") << "[DTVDriftWriter]Constructor called!";

  if(granularity_ != "mySL")
     throw cms::Exception("Configuration") << "[DTVDriftWriter] Check parameter calibGranularity: " << granularity_ << " option not available.";

  //mTimeMap_ = new DTMtime();

  // Get the concrete algo from the factory
  string algoName = pset.getParameter<string>("vDriftAlgo");
  ParameterSet algoPSet = pset.getParameter<ParameterSet>("vDriftAlgoConfig");
  vDriftAlgo_ = DTVDriftPluginFactory::get()->create(algoName,algoPSet);
}

DTVDriftWriter::~DTVDriftWriter(){
  LogVerbatim("Calibration") << "[DTVDriftWriter]Destructor called!";
  delete vDriftAlgo_;
}

void DTVDriftWriter::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  // Get geometry from Event Setup
  setup.get<MuonGeometryRecord>().get(dtGeom_);
  // Pass EventSetup to concrete implementation
  vDriftAlgo_->setES(setup); 
}

void DTVDriftWriter::endJob() {
  DTMtime* mTimeMap = new DTMtime();

  if(granularity_ == "bySL") {    
     // Get all the sls from the geometry
     const vector<DTSuperLayer*>& superLayers = dtGeom_->superLayers(); 
     vector<DTSuperLayer*>::const_iterator sl = superLayers.begin();
     vector<DTSuperLayer*>::const_iterator sl_end = superLayers.end();
     for(; sl != sl_end; ++sl){
        DTSuperLayerId slId = (*sl)->id();
        // Compute vDrift
        try{
           DTVDriftData vDriftData = vDriftAlgo_->compute(slId);
           double vDriftMean = vDriftData.mean;
           double vDriftSigma = vDriftData.sigma; 
           // vdrift is cm/ns , resolution is cm
           mTimeMap->set(slId,
                         vDriftMean,
		         vDriftSigma,
		         DTVelocityUnits::cm_per_ns);
           LogVerbatim("Calibration") << "vDrift for: " << slId
                                      << " Mean " << vDriftMean
                                      << " Sigma " << vDriftSigma;
        } catch(cms::Exception& e){
           LogError("Calibration") << e.explainSelf();
           mTimeMap->set(slId,
                         vDriftDef_,
                         vDriftResoDef_,
                         DTVelocityUnits::cm_per_ns);
           LogVerbatim("Calibration") << "Using fake vDrift for: " << slId
                                      << " Mean " << vDriftDef_
                                      << " Sigma " << vDriftResoDef_;
        }
     } // End of loop on superlayers
  }

  // Write the vDrift object to DB
  LogVerbatim("Calibration") << "[DTVDriftWriter]Writing vdrift object to DB!";
  string record = "DTMtimeRcd";
  DTCalibDBUtils::writeToDB<DTMtime>(record, mTimeMap);
}
