
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/02 19:47:33 $
 *  $Revision: 1.12 $
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

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"
#include "CalibMuon/DTCalibration/interface/DTVDriftPluginFactory.h"
#include "CalibMuon/DTCalibration/interface/DTVDriftBaseAlgo.h"

#include <string>
#include <vector>

using namespace std;
using namespace edm;

DTVDriftWriter::DTVDriftWriter(const ParameterSet& pset):
  granularity_( pset.getUntrackedParameter<string>("calibGranularity","bySL") ) {

  LogVerbatim("Calibration") << "[DTVDriftWriter]Constructor called!";

  if(granularity_ != "bySL")
     throw cms::Exception("Configuration") << "[DTVDriftWriter] Check parameter calibGranularity: " << granularity_ << " option not available.";

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
  // Get the map of ttrig from the Setup
  ESHandle<DTMtime> mTime;
  setup.get<DTMtimeRcd>().get(mTime);
  mTimeMap_ = &*mTime;

  // Get geometry from Event Setup
  setup.get<MuonGeometryRecord>().get(dtGeom_);
  // Pass EventSetup to concrete implementation
  vDriftAlgo_->setES(setup); 
}

void DTVDriftWriter::endJob() {
  // Create the object to be written to DB
  DTMtime* mTimeNewMap = new DTMtime();

  if(granularity_ == "bySL") {    
     // Get all the sls from the geometry
     const vector<DTSuperLayer*>& superLayers = dtGeom_->superLayers(); 
     vector<DTSuperLayer*>::const_iterator sl = superLayers.begin();
     vector<DTSuperLayer*>::const_iterator sl_end = superLayers.end();
     for(; sl != sl_end; ++sl){
        DTSuperLayerId slId = (*sl)->id();
        // Get original value from DB
        float vDrift = 0., resolution = 0.;
        // vdrift is cm/ns , resolution is cm
        int status = mTimeMap_->get(slId,vDrift,resolution,DTVelocityUnits::cm_per_ns);

        // Compute vDrift
        try{
           dtCalibration::DTVDriftData vDriftData = vDriftAlgo_->compute(slId);
           float vDriftNew = vDriftData.vdrift;
           float resolutionNew = vDriftData.resolution; 
           // vdrift is cm/ns , resolution is cm
           mTimeNewMap->set(slId,
                            vDriftNew,
		            resolutionNew,
		            DTVelocityUnits::cm_per_ns);
           LogVerbatim("Calibration") << "vDrift for: " << slId
                                      << " Mean " << vDriftNew
                                      << " Resolution " << resolutionNew;
        } catch(cms::Exception& e){
           LogError("Calibration") << e.explainSelf();
           // Go back to original value in case of error
           if(!status){  
              mTimeNewMap->set(slId,
                               vDrift,
                               resolution,
                               DTVelocityUnits::cm_per_ns);
              LogVerbatim("Calibration") << "Keep original vDrift for: " << slId
                                         << " Mean " << vDrift
                                         << " Resolution " << resolution;
           }
        }
     } // End of loop on superlayers
  }

  // Write the vDrift object to DB
  LogVerbatim("Calibration") << "[DTVDriftWriter]Writing vdrift object to DB!";
  string record = "DTMtimeRcd";
  DTCalibDBUtils::writeToDB<DTMtime>(record, mTimeNewMap);
}
