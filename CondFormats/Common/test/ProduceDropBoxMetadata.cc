
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

#include "CondFormats/Common/src/headers.h"

#include <iostream>

using namespace std;
using namespace edm;

ProduceDropBoxMetadata::ProduceDropBoxMetadata(const edm::ParameterSet& pSet) {
  read = pSet.getUntrackedParameter<bool>("read");
  write = pSet.getUntrackedParameter<bool>("write");

  fToWrite = pSet.getParameter<vector<ParameterSet> >("toWrite");
  fToRead = pSet.getUntrackedParameter<vector<string> >("toRead");
}

ProduceDropBoxMetadata::~ProduceDropBoxMetadata() {}

// void ProduceDropBoxMetadata::beginJob() {
void ProduceDropBoxMetadata::beginRun(const edm::Run& run, const edm::EventSetup& eSetup) {
  cout << "[ProduceDropBoxMetadata] beginJob" << endl;

  string plRecord = "DropBoxMetadataRcd";
  // ---------------------------------------------------------------------------------
  // Write the payload

  if (write) {
    cout << "\n\n[ProduceDropBoxMetadata] entering write, to loop over toWrite\n" << endl;
    DropBoxMetadata* metadata = new DropBoxMetadata;

    // loop over all the pSets for the TF1 that we want to write to DB
    for (const auto& fSetup : fToWrite) {
      string record = fSetup.getUntrackedParameter<string>("record");
      cout << "\n--- record: " << record << endl;
      DropBoxMetadata::Parameters params;
      vector<string> paramKeys = fSetup.getParameterNames();
      for (const auto& paramKey : paramKeys) {
        if (paramKey != "record") {
          string value = fSetup.getUntrackedParameter<string>(paramKey);
          params.addParameter(paramKey, value);
          cout << "           key: " << paramKey << " value: " << value << endl;
        }
      }
      metadata->addRecordParameters(record, params);
    }

    // actually write to DB
    edm::Service<cond::service::PoolDBOutputService> dbOut;
    if (dbOut.isAvailable()) {
      dbOut->writeOne<DropBoxMetadata>(metadata, 1, plRecord);
    }
  }

  if (read) {
    // Read the objects
    cout << "\n\n[ProduceDropBoxMetadata] entering read, to loop over toRead\n" << endl;
    edm::ESHandle<DropBoxMetadata> mdPayload;
    eSetup.get<DropBoxMetadataRcd>().get(mdPayload);

    const DropBoxMetadata* metadata = mdPayload.product();

    // loop
    for (const auto& name : fToRead) {
      cout << "\n--- record: " << name << endl;
      if (metadata->knowsRecord(name)) {
        const map<string, string>& params = metadata->getRecordParameters(name).getParameterMap();
        for (const auto& param : params) {
          cout << "           key: " << param.first << " value: " << param.second << endl;
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
