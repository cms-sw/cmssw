
/** \class ProduceDropBoxMetadata
 *  Create an sqlite file containing the 
 *  DropBoxMetadata needed for the PCL uploads
 *
 *  $Date: 2011/02/22 11:05:16 $
 *  $Revision: 1.4 $
 *  \author G. Cerminara - CERN
 *  Modifications T. Vami - JHU
 */

#include <iostream>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/DropBoxMetadataRcd.h"
#include "CondFormats/Common/interface/DropBoxMetadata.h"
#include "CondFormats/Common/src/headers.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class ProduceDropBoxMetadata : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  explicit ProduceDropBoxMetadata(const edm::ParameterSet&);

  /// Destructor
  ~ProduceDropBoxMetadata() override;

private:
  // Operations
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  bool read;
  bool write;

  std::vector<edm::ParameterSet> fToWrite;
  std::vector<std::string> fToRead;

  const edm::ESGetToken<DropBoxMetadata, DropBoxMetadataRcd> dropBoxMetadataToken_;
};

using namespace std;
using namespace edm;

ProduceDropBoxMetadata::ProduceDropBoxMetadata(const edm::ParameterSet& pSet) : dropBoxMetadataToken_(esConsumes()) {
  read = pSet.getUntrackedParameter<bool>("read");
  write = pSet.getUntrackedParameter<bool>("write");

  fToWrite = pSet.getParameter<vector<ParameterSet> >("toWrite");
  fToRead = pSet.getUntrackedParameter<vector<string> >("toRead");
}

ProduceDropBoxMetadata::~ProduceDropBoxMetadata() = default;

void ProduceDropBoxMetadata::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // ---------------------------------------------------------------------------------
  // Write the payload
  if (write) {
    DropBoxMetadata metadata;

    edm::LogPrint("ProduceDropBoxMetadata") << "Entering write, to loop over toWrite";
    // loop over all the pSets for the TF1 that we want to write to DB
    for (vector<ParameterSet>::const_iterator fSetup = fToWrite.begin(); fSetup != fToWrite.end(); ++fSetup) {
      string record = (*fSetup).getUntrackedParameter<string>("record");
      edm::LogPrint("ProduceDropBoxMetadata") << "\n--- record: " << record;
      DropBoxMetadata::Parameters params;
      vector<string> paramKeys = (*fSetup).getParameterNames();
      for (vector<string>::const_iterator key = paramKeys.begin(); key != paramKeys.end(); ++key) {
        if (*key != "record") {
          string value = (*fSetup).getUntrackedParameter<string>(*key);
          params.addParameter(*key, value);
          edm::LogPrint("ProduceDropBoxMetadata") << "           key: " << *key << " value: " << value;
        }
      }
      metadata.addRecordParameters(record, params);
    }

    // actually write to DB
    edm::Service<cond::service::PoolDBOutputService> dbOut;
    if (dbOut.isAvailable()) {
      dbOut->writeOneIOV<DropBoxMetadata>(metadata, 1, "DropBoxMetadataRcd");
    }
  }

  if (read) {
    // Read the objects
    edm::LogPrint("ProduceDropBoxMetadata") << "Entering read, to loop over toRead";
    const auto& mdPayload = iSetup.getData(dropBoxMetadataToken_);

    // loop
    for (vector<string>::const_iterator name = fToRead.begin(); name != fToRead.end(); ++name) {
      edm::LogPrint("ProduceDropBoxMetadata") << "\n--- record: " << *name;
      if (mdPayload.knowsRecord(*name)) {
        const map<string, string>& params = mdPayload.getRecordParameters(*name).getParameterMap();
        for (map<string, string>::const_iterator par = params.begin(); par != params.end(); ++par) {
          edm::LogPrint("ProduceDropBoxMetadata") << "           key: " << par->first << " value: " << par->second;
        }
      } else {
        edm::LogPrint("ProduceDropBoxMetadata") << "     not in the payload!";
      }
    }
  }
}

DEFINE_FWK_MODULE(ProduceDropBoxMetadata);
