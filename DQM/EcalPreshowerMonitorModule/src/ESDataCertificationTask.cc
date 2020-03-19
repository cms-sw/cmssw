#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESDataCertificationTask.h"

using namespace cms;
using namespace edm;
using namespace std;

ESDataCertificationTask::ESDataCertificationTask(const ParameterSet& ps) {
  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meESDataCertificationSummary_ = nullptr;
  meESDataCertificationSummaryMap_ = nullptr;
}

ESDataCertificationTask::~ESDataCertificationTask() {}

void ESDataCertificationTask::beginJob(void) {
  char histo[200];

  if (dqmStore_) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    sprintf(histo, "CertificationSummary");
    meESDataCertificationSummary_ = dqmStore_->bookFloat(histo);
    meESDataCertificationSummary_->Fill(0.0);

    sprintf(histo, "CertificationSummaryMap");
    meESDataCertificationSummaryMap_ = dqmStore_->book2D(histo, histo, 40, 0., 40., 40, 0., 40.);
    meESDataCertificationSummaryMap_->setAxisTitle("X", 1);
    meESDataCertificationSummaryMap_->setAxisTitle("Y", 2);
  }
}

void ESDataCertificationTask::endJob(void) {}

void ESDataCertificationTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
                                                   const edm::EventSetup& iSetup) {
  this->reset();
}

void ESDataCertificationTask::reset(void) {
  if (meESDataCertificationSummary_)
    meESDataCertificationSummary_->Reset();

  if (meESDataCertificationSummaryMap_)
    meESDataCertificationSummaryMap_->Reset();
}

void ESDataCertificationTask::analyze(const Event& e, const EventSetup& c) {}

DEFINE_FWK_MODULE(ESDataCertificationTask);
