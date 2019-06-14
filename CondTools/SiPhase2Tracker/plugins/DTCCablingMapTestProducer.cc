// -*- C++ -*-
//
// Package:    CondTools/SiPhase2Tracker
// Class:      DTCCablingMapTestProducer
//
/**\class DTCCablingMapTestProducer DTCCablingMapTestProducer.cc CondTools/SiPhase2Tracker/plugins/DTCCablingMapTestProducer.cc

Description: [one line class summary]

Implementation:
		[Notes on implementation]
*/
//
// Original Author:  Luigi Calligaris, SPRACE, São Paulo, BR
// Created        :  Wed, 27 Feb 2019 21:41:13 GMT
//
//

#include <memory>

#include <unordered_map>
#include <utility>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"

class DTCCablingMapTestProducer : public edm::one::EDAnalyzer<> {
public:
  explicit DTCCablingMapTestProducer(const edm::ParameterSet&);
  ~DTCCablingMapTestProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  cond::Time_t iovBeginTime_;
  std::unique_ptr<TrackerDetToDTCELinkCablingMap> pCablingMap_;
  std::string recordName_;
};

void DTCCablingMapTestProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Stores a dummy TrackerDetToDTCELinkCablingMap database object from a CSV file.");
  desc.add<long long unsigned int>("iovBeginTime", 1);
  desc.add<std::string>("record", "TrackerDetToDTCELinkCablingMap");
  descriptions.add("DTCCablingMapTestProducer", desc);
}

DTCCablingMapTestProducer::DTCCablingMapTestProducer(const edm::ParameterSet& iConfig)
    : iovBeginTime_(iConfig.getParameter<long long unsigned int>("iovBeginTime")),
      pCablingMap_(std::make_unique<TrackerDetToDTCELinkCablingMap>()),
      recordName_(iConfig.getParameter<std::string>("record")) {}

void DTCCablingMapTestProducer::beginJob() {
  using namespace edm;
  using namespace std;

  pCablingMap_->insert(DTCELinkId(101u, 1u, 2u), 11111111);
  pCablingMap_->insert(DTCELinkId(102u, 2u, 2u), 22222222);
  pCablingMap_->insert(DTCELinkId(103u, 3u, 3u), 33333333);
  pCablingMap_->insert(DTCELinkId(104u, 4u, 4u), 44444444);

  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable())
    poolDbService->writeOne(pCablingMap_.release(), iovBeginTime_, recordName_);
  else
    throw std::runtime_error("PoolDBService required.");
}

void DTCCablingMapTestProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

void DTCCablingMapTestProducer::endJob() {}

DTCCablingMapTestProducer::~DTCCablingMapTestProducer() {}

//define this as a plug-in
DEFINE_FWK_MODULE(DTCCablingMapTestProducer);
