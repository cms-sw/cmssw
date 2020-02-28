#include "DPGAnalysis/Skims/interface/DetStatus.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

//
// -- Constructor
//
DetStatus::DetStatus(const edm::ParameterSet& pset) {
  verbose_ = pset.getUntrackedParameter<bool>("DebugOn", false);
  AndOr_ = pset.getParameter<bool>("AndOr");
  applyfilter_ = pset.getParameter<bool>("ApplyFilter");
  DetNames_ = pset.getParameter<std::vector<std::string> >("DetectorType");

  // build a map
  DetMap_ = 0;

  for (unsigned int detreq = 0; detreq < DetNames_.size(); detreq++) {
    for (unsigned int detlist = 0; detlist < DcsStatus::nPartitions; detlist++) {
      if (DetNames_[detreq] == DcsStatus::partitionName[detlist]) {
        //edm::LogPrint("DetStatus") << __PRETTY_FUNCTION__ << " requested:" << DetNames_[detreq] << std::endl;

        // set for DCSRecord
        requestedPartitions_.set(detlist, true);

        // set for DCSStatus
        DetMap_ |= (1 << DcsStatus::partitionList[detlist]);

        if (verbose_) {
          edm::LogInfo("DetStatus") << "DCSStatus filter: asked partition " << DcsStatus::partitionName[detlist]
                                    << " bit " << DcsStatus::partitionList[detlist] << std::endl;
        }
      }
    }
  }
  scalersToken_ = consumes<DcsStatusCollection>(edm::InputTag("scalersRawToDigi"));
  dcsRecordToken_ = consumes<DCSRecord>(edm::InputTag("onlineMetaDataDigis"));
}

//
// -- Destructor
//
DetStatus::~DetStatus() {}

bool DetStatus::checkForDCSStatus(const DcsStatusCollection& dcsStatus) {
  if (verbose_) {
    edm::LogInfo("DetStatus") << "Using FED#735 for reading DCS bits" << std::endl;
  }

  bool accepted = false;
  unsigned int curr_dcs = (dcsStatus)[0].ready();
  if (verbose_) {
    edm::LogVerbatim("DetStatus") << "curr_dcs = " << curr_dcs << std::endl;
  }

  if (AndOr_)
    accepted = ((DetMap_ & curr_dcs) == DetMap_);
  else
    accepted = ((DetMap_ & curr_dcs) != 0);

  if (verbose_) {
    edm::LogInfo("DetStatus") << "DCSStatus filter: requested map: " << DetMap_ << " dcs in event: " << curr_dcs
                              << " filter: " << accepted << "( AndOr: " << AndOr_ << ")" << std::endl;
    edm::LogVerbatim("DetStatus") << "Partitions ON: ";
    for (unsigned int detlist = 0; detlist < DcsStatus::nPartitions; detlist++) {
      if ((dcsStatus)[0].ready(DcsStatus::partitionList[detlist])) {
        edm::LogVerbatim("DetStatus") << " " << DcsStatus::partitionName[detlist];
      }
    }
    edm::LogVerbatim("DetStatus") << std::endl;
  }
  return accepted;
}

bool DetStatus::checkForDCSRecord(const DCSRecord& dcsRecord) {
  bool accepted = false;

  if (verbose_) {
    edm::LogInfo("DetStatus") << "Using softFED#1022 for reading DCS bits" << std::endl;
  }

  for (unsigned int detlist = 0; detlist < DcsStatus::nPartitions; detlist++) {
    if (requestedPartitions_.test(detlist)) {
      if (AndOr_) {
        accepted = (accepted & dcsRecord.highVoltageReady(detlist));
      } else {
        accepted = (accepted || dcsRecord.highVoltageReady(detlist));
      }
    }
  }

  if (verbose_) {
    edm::LogInfo("DetStatus") << "DCSStatus filter: " << accepted << "( AndOr: " << AndOr_ << ")" << std::endl;
    edm::LogVerbatim("DetStatus") << "Partitions ON: ";
    for (unsigned int detlist = 0; detlist < DcsStatus::nPartitions; detlist++) {
      if ((dcsRecord.highVoltageReady(detlist))) {
        edm::LogVerbatim("DetStatus") << " " << DcsStatus::partitionName[detlist];
      }
    }
    edm::LogVerbatim("DetStatus") << std::endl;
  }

  return accepted;
}

bool DetStatus::filter(edm::Event& evt, edm::EventSetup const& es) {
  bool accepted = false;

  DcsStatusCollection const& dcsStatus = evt.get(scalersToken_);
  DCSRecord const& dcsRecord = evt.get(dcsRecordToken_);

  if (!dcsStatus.empty()) {
    accepted = checkForDCSStatus(dcsStatus);
  } else {
    accepted = checkForDCSRecord(dcsRecord);
  }

  if (!applyfilter_) {
    accepted = true;
  }

  return accepted;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DetStatus);
