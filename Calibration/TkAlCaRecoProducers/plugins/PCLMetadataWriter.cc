
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - CERN
 */

#include "Calibration/TkAlCaRecoProducers/plugins/PCLMetadataWriter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/MessageLogger/interface/JobReport.h"


using namespace std;
using namespace edm;

PCLMetadataWriter::PCLMetadataWriter(const edm::ParameterSet& pSet){


  vector<ParameterSet> recordsToMap = pSet.getParameter<vector<ParameterSet> >("recordsToMap");
  for(vector<ParameterSet>::const_iterator recordPset = recordsToMap.begin();
      recordPset != recordsToMap.end();
      ++recordPset) {
    
    string record = (*recordPset).getUntrackedParameter<string>("record");
    
    map<string, string> jrInfo;
    jrInfo["Source"] = "AlcaHarvesting";
    jrInfo["FileClass"] = "ALCA";
    jrInfo["destDB"] = (*recordPset).getUntrackedParameter<string>("destDB");
    jrInfo["destDBValidation"] = (*recordPset).getUntrackedParameter<string>("destDBValidation");
    jrInfo["tag"] = (*recordPset).getUntrackedParameter<string>("tag");
    jrInfo["Timetype"] = (*recordPset).getUntrackedParameter<string>("Timetype");
    jrInfo["IOVCheck"] = (*recordPset).getUntrackedParameter<string>("IOVCheck");
    jrInfo["DuplicateTagHLT"] = (*recordPset).getUntrackedParameter<string>("DuplicateTagHLT");
    jrInfo["DuplicateTagEXPRESS"] = (*recordPset).getUntrackedParameter<string>("DuplicateTagEXPRESS");
    jrInfo["DuplicateTagPROMPT"] = (*recordPset).getUntrackedParameter<string>("DuplicateTagPROMPT");

    recordMap[record] = jrInfo;

  }

}

PCLMetadataWriter::~PCLMetadataWriter(){}


void PCLMetadataWriter::analyze(const edm::Event& event, const edm::EventSetup& eSetup) {}


void PCLMetadataWriter::beginRun(const edm::Run& run, const edm::EventSetup& eSetup) {} 

void PCLMetadataWriter::endRun(const edm::Run& run, const edm::EventSetup& eSetup) {
  
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
	jrInfo["inputtag"] = poolDbService->tag(record);

	
	// actually write in the job report
	jr->reportAnalysisFile(filename, jrInfo);
      }
    }
  }
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PCLMetadataWriter);
