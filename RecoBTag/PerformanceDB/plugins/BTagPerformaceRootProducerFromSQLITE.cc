// -*- C++ -*-
//
// Package:    BTagPerformaceRootProducerFromSQLITE
// Class:      BTagPerformaceRootProducerFromSQLITE
//
/**\class BTagPerformaceRootProducerFromSQLITE BTagPerformaceRootProducerFromSQLITE.cc junk/BTagPerformaceRootProducerFromSQLITE/src/BTagPerformaceRootProducerFromSQLITE.cc

 Description: This writes out a ROOT file with a BtagPerformance object taken from an sqlite file. 

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Thu Feb 11 14:21:59 CST 2010
//
//

// system include files
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "DataFormats/Provenance/interface/ESRecordAuxiliary.h"
#include "DataFormats/FWLite/interface/format_type_name.h"

#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"
#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"

#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"

//
// class declaration
//

class BTagPerformaceRootProducerFromSQLITE : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BTagPerformaceRootProducerFromSQLITE(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  std::vector<std::string> names_;
  std::vector<edm::ESGetToken<BtagPerformance, BTagPerformanceRecord>> tokens_;
  edm::ESWatcher<BTagPerformanceRecord> recWatcher_;
  std::unique_ptr<fwlite::RecordWriter> writer_;
  edm::IOVSyncValue lastValue_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BTagPerformaceRootProducerFromSQLITE::BTagPerformaceRootProducerFromSQLITE(const edm::ParameterSet& iConfig)
    : names_(iConfig.getParameter<std::vector<std::string>>("names")) {
  usesResource(TFileService::kSharedResource);
  tokens_.reserve(names_.size());
  for (auto const& n : names_) {
    tokens_.push_back(esConsumes<BtagPerformance, BTagPerformanceRecord>(edm::ESInputTag("", n)));
  }
}

//
// member functions
//

// ------------ method called to for each event  ------------
void BTagPerformaceRootProducerFromSQLITE::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("BTagPerformaceRootProducerFromSQLITE") << "Hello from BTagPerformaceRootProducerFromSQLITE!";
  if (recWatcher_.check(iSetup)) {
    const BTagPerformanceRecord& r = iSetup.get<BTagPerformanceRecord>();

    if (!writer_.get()) {
      edm::Service<TFileService> fs;
      TFile* f = &(fs->file());
      writer_ = std::make_unique<fwlite::RecordWriter>(r.key().name(), f);
    }
    lastValue_ = r.validityInterval().last();

    for (size_t i = 0; i < names_.size(); i++) {
      edm::LogInfo("BTagPerformaceRootProducerFromSQLITE") << " Studying performance with label " << names_[i];
      const BtagPerformance& perf = r.get(tokens_[i]);

      writer_->update(&(perf.payload()), typeid(PerformancePayload), names_[i].c_str());
      writer_->update(&(perf.workingPoint()), typeid(PerformanceWorkingPoint), names_[i].c_str());
    }
    writer_->fill(edm::ESRecordAuxiliary(r.validityInterval().first().eventID(), r.validityInterval().first().time()));
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformaceRootProducerFromSQLITE);
