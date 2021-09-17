#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESDcsInfoTask.h"

using namespace cms;
using namespace edm;
using namespace std;

ESDcsInfoTask::ESDcsInfoTask(const ParameterSet& ps) {
  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  dcsStatustoken_ = consumes<DcsStatusCollection>(ps.getParameter<InputTag>("DcsStatusLabel"));

  meESDcsFraction_ = nullptr;
  meESDcsActiveMap_ = nullptr;
}

ESDcsInfoTask::~ESDcsInfoTask() {}

void ESDcsInfoTask::beginJob(void) {
  char histo[200];

  if (dqmStore_) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    sprintf(histo, "DCSSummary");
    meESDcsFraction_ = dqmStore_->bookFloat(histo);
    meESDcsFraction_->Fill(-1.0);

    sprintf(histo, "DCSSummaryMap");
    meESDcsActiveMap_ = dqmStore_->book1D(histo, histo, 2, 0., 2.);
    meESDcsActiveMap_->setAxisTitle("(ES+/ES-)", 1);
  }
}

void ESDcsInfoTask::endJob(void) {}

void ESDcsInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup) {
  this->reset();

  for (int i = 0; i < 2; i++) {
    meESDcsActiveMap_->setBinContent(i + 1, -1.0);
  }
}

void ESDcsInfoTask::reset(void) {
  if (meESDcsFraction_)
    meESDcsFraction_->Reset();

  if (meESDcsActiveMap_)
    meESDcsActiveMap_->Reset();
}

void ESDcsInfoTask::analyze(const Event& e, const EventSetup& c) {
  ievt_++;

  float ESpDcsStatus = 0;
  float ESmDcsStatus = 0;

  Handle<DcsStatusCollection> dcsStatus;
  e.getByToken(dcsStatustoken_, dcsStatus);
  if (dcsStatus.isValid()) {
    for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); dcsStatusItr != dcsStatus->end();
         ++dcsStatusItr) {
      ESpDcsStatus = dcsStatusItr->ready(DcsStatus::ESp);
      ESmDcsStatus = dcsStatusItr->ready(DcsStatus::ESm);
    }

    ESpDcsStatus = (ESpDcsStatus + float(ievt_ - 1) * meESDcsActiveMap_->getBinContent(1)) / float(ievt_);
    ESmDcsStatus = (ESmDcsStatus + float(ievt_ - 1) * meESDcsActiveMap_->getBinContent(2)) / float(ievt_);
  }

  meESDcsActiveMap_->setBinContent(1, ESpDcsStatus);
  meESDcsActiveMap_->setBinContent(2, ESmDcsStatus);

  meESDcsFraction_->Fill((ESpDcsStatus + ESmDcsStatus) / 2.);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ESDcsInfoTask);
