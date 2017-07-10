
/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - CERN
 */

#include "Calibration/TkAlCaRecoProducers/plugins/PCLMetadataWriter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/DropBoxMetadataRcd.h"
#include "CondFormats/Common/interface/DropBoxMetadata.h"

#include <iostream>

using namespace std;
using namespace edm;

PCLMetadataWriter::PCLMetadataWriter(const edm::ParameterSet& pSet){
  
  readFromDB = pSet.getParameter<bool>("readFromDB");
  

  vector<ParameterSet> recordsToMap = pSet.getParameter<vector<ParameterSet> >("recordsToMap");
  for(vector<ParameterSet>::const_iterator recordPset = recordsToMap.begin();
      recordPset != recordsToMap.end();
      ++recordPset) {

    // record is the key which identifies one set of metadata in DropBoxMetadataRcd
    // (not necessarily a record in the strict framework sense)
    string record = (*recordPset).getUntrackedParameter<string>("record");

    map<string, string> jrInfo;
    if(!readFromDB) {
      vector<string> paramKeys = (*recordPset).getParameterNames();
      for(vector<string>::const_iterator key = paramKeys.begin();
	  key != paramKeys.end();
	  ++key) {
	jrInfo["Source"] = "AlcaHarvesting";
	jrInfo["FileClass"] = "ALCA";
	if(*key != "record") {
	  jrInfo[*key] = (*recordPset).getUntrackedParameter<string>(*key);
	}
      }
    }
    recordMap[record] = jrInfo;

  }

}

PCLMetadataWriter::~PCLMetadataWriter(){}


void PCLMetadataWriter::analyze(const edm::Event& event, const edm::EventSetup& eSetup) {}


void PCLMetadataWriter::beginRun(const edm::Run& run, const edm::EventSetup& eSetup) {} 

void PCLMetadataWriter::endRun(const edm::Run& run, const edm::EventSetup& eSetup) {

  const DropBoxMetadata *metadata = 0;

  if(readFromDB) {
    // Read the objects
    edm::ESHandle<DropBoxMetadata> mdPayload;
    eSetup.get<DropBoxMetadataRcd>().get(mdPayload);
    
    metadata = mdPayload.product();
  }

  // get the PoolDBOutputService
  Service<cond::service::PoolDBOutputService> poolDbService;
  if(poolDbService.isAvailable() ) {
    edm::Service<edm::JobReport> jr;
    if (jr.isAvailable()) {
      // the filename is unique for all records
      string filename = poolDbService->session().connectionString();

      // loop over all records
      for(map<string,  map<string, string> >::const_iterator recordAndMap = recordMap.begin();
	  recordAndMap != recordMap.end();
	  ++recordAndMap) {

	string record = (*recordAndMap).first;

	// this is the map of metadata that we write in the JR
	map<string, string> jrInfo = (*recordAndMap).second;
	if(readFromDB) {
	  if(metadata->knowsRecord(record)) {
	    jrInfo = metadata->getRecordParameters(record).getParameterMap();
	  }
	}
	
	// name of the the input tag in the metadata for the condUploader metadata
	// needs to be the same as the tag written out by the harvesting step
	jrInfo["inputtag"] = poolDbService->tag(record);
	
	// actually write in the job report
	jr->reportAnalysisFile(filename, jrInfo);

      }
    }
  }
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PCLMetadataWriter);
