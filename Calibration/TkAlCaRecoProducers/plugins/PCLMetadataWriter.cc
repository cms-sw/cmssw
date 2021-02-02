/** \class PCLMetadataWriter
 *  No description available.
 *
 *  \author G. Cerminara - CERN
 */

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Common/interface/DropBoxMetadata.h"
#include "CondFormats/DataRecord/interface/DropBoxMetadataRcd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>
#include <vector>
#include <iostream>

class PCLMetadataWriter : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  PCLMetadataWriter(const edm::ParameterSet &);

  /// Destructor
  ~PCLMetadataWriter() override = default;

  // Operations
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginRun(const edm::Run &, const edm::EventSetup &);
  void endRun(const edm::Run &, const edm::EventSetup &);

protected:
private:
  const edm::ESGetToken<DropBoxMetadata, DropBoxMetadataRcd> dropBoxToken_;
  bool readFromDB;
  std::map<std::string, std::map<std::string, std::string>> recordMap;
};

using namespace std;
using namespace edm;

PCLMetadataWriter::PCLMetadataWriter(const edm::ParameterSet &pSet)
    : dropBoxToken_(esConsumes<DropBoxMetadata, DropBoxMetadataRcd, edm::Transition::EndRun>()) {
  readFromDB = pSet.getParameter<bool>("readFromDB");

  vector<ParameterSet> recordsToMap = pSet.getParameter<vector<ParameterSet>>("recordsToMap");
  for (vector<ParameterSet>::const_iterator recordPset = recordsToMap.begin(); recordPset != recordsToMap.end();
       ++recordPset) {
    // record is the key which identifies one set of metadata in
    // DropBoxMetadataRcd (not necessarily a record in the strict framework
    // sense)
    string record = (*recordPset).getUntrackedParameter<string>("record");

    map<string, string> jrInfo;
    if (!readFromDB) {
      vector<string> paramKeys = (*recordPset).getParameterNames();
      for (vector<string>::const_iterator key = paramKeys.begin(); key != paramKeys.end(); ++key) {
        jrInfo["Source"] = "AlcaHarvesting";
        jrInfo["FileClass"] = "ALCA";
        if (*key != "record") {
          jrInfo[*key] = (*recordPset).getUntrackedParameter<string>(*key);
        }
      }
    }
    recordMap[record] = jrInfo;
  }
}

void PCLMetadataWriter::analyze(const edm::Event &event, const edm::EventSetup &eSetup) {}

void PCLMetadataWriter::beginRun(const edm::Run &run, const edm::EventSetup &eSetup) {}

void PCLMetadataWriter::endRun(const edm::Run &run, const edm::EventSetup &eSetup) {
  const DropBoxMetadata *metadata = nullptr;

  if (readFromDB) {
    // Read the objects
    metadata = &eSetup.getData(dropBoxToken_);
  }

  // get the PoolDBOutputService
  Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    edm::Service<edm::JobReport> jr;
    if (jr.isAvailable()) {
      // the filename is unique for all records
      string filename = poolDbService->session().connectionString();

      // loop over all records
      for (map<string, map<string, string>>::const_iterator recordAndMap = recordMap.begin();
           recordAndMap != recordMap.end();
           ++recordAndMap) {
        string record = (*recordAndMap).first;

        // this is the map of metadata that we write in the JR
        map<string, string> jrInfo = (*recordAndMap).second;
        if (readFromDB) {
          if (metadata->knowsRecord(record)) {
            jrInfo = metadata->getRecordParameters(record).getParameterMap();
          }
        }

        // name of the the input tag in the metadata for the condUploader
        // metadata needs to be the same as the tag written out by the
        // harvesting step
        jrInfo["inputtag"] = poolDbService->tag(record);

        // actually write in the job report
        jr->reportAnalysisFile(filename, jrInfo);
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PCLMetadataWriter);
