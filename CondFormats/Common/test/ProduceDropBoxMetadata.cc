
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/02/15 14:07:10 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - CERN
 */

#include "ProduceDropBoxMetadata.h"

#include <iostream>
#include <vector>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/DropBoxMetadataRcd.h"
#include "CondFormats/Common/interface/DropBoxMetadata.h"


#include <iostream>


using namespace std;
using namespace edm;

ProduceDropBoxMetadata::ProduceDropBoxMetadata(const edm::ParameterSet& pSet) {
  

  read = pSet.getUntrackedParameter<bool>("read");
  write = pSet.getUntrackedParameter<bool>("write");

  fToWrite =  pSet.getParameter<vector<ParameterSet> >("toWrite");
  fToRead =  pSet.getUntrackedParameter<vector<string> >("toRead");

}

ProduceDropBoxMetadata::~ProduceDropBoxMetadata(){}




// void ProduceDropBoxMetadata::beginJob() {
void ProduceDropBoxMetadata::beginRun(const edm::Run& run, const edm::EventSetup& eSetup) {

  cout << "[ProduceDropBoxMetadata] beginJob" << endl;



  string plRecord = "DropBoxMetadataRcd";
  // ---------------------------------------------------------------------------------
  // Write the payload

 

  if(write) {
    
    DropBoxMetadata *metadata = new DropBoxMetadata;

    // loop over all the pSets for the TF1 that we want to write to DB
    for(vector<ParameterSet>::const_iterator fSetup = fToWrite.begin();
	fSetup != fToWrite.end();
	++fSetup) {
      
      string record = (*fSetup).getUntrackedParameter<string>("record");
      cout << "--- record: " << record << endl;
      DropBoxMetadata::Parameters params;
      vector<string> paramKeys = (*fSetup).getParameterNames();
      for(vector<string>::const_iterator key = paramKeys.begin();
	  key != paramKeys.end();
	  ++key) {
	if(*key != "record") {
	  string value = (*fSetup).getUntrackedParameter<string>(*key);
	  params.addParameter(*key, value);
	  cout << "           key: " << *key << " value: " << value << endl;
	}
      }
      metadata->addRecordParameters(record, params);
    }


    // actually write to DB
    edm::Service<cond::service::PoolDBOutputService> dbOut;
    if(dbOut.isAvailable()) {
      dbOut->writeOne<DropBoxMetadata>(metadata,  1, plRecord);
    }
  }


  if(read) {
    // Read the objects
    edm::ESHandle<DropBoxMetadata> mdPayload;
    eSetup.get<DropBoxMetadataRcd>().get(mdPayload);

    const DropBoxMetadata *metadata = mdPayload.product();
      
    // loop 
    for(vector<string>::const_iterator name = fToRead.begin();
	name != fToRead.end(); ++name) {
      cout << "--- record: " << *name << endl;
      if(metadata->knowsRecord(*name)) {
	const map<string, string>  & params = metadata->getRecordParameters(*name).getParameterMap();
	for(map<string, string>::const_iterator par = params.begin();
	    par != params.end(); ++ par) {
	  cout << "           key: " << par->first << " value: " << par->second << endl;
	}
      } else {
	cout << "     not in the payload!" << endl;
      }
    }

  }



}

     
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ProduceDropBoxMetadata);
