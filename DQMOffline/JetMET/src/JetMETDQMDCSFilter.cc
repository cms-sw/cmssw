#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// -- Constructor
//
JetMETDQMDCSFilter::JetMETDQMDCSFilter(const edm::ParameterSet& pset, edm::ConsumesCollector& iC) {
  verbose_ = pset.getUntrackedParameter<bool>("DebugOn", false);
  detectorTypes_ = pset.getUntrackedParameter<std::string>("DetectorTypes", "ecal:hcal");
  filter_ = !pset.getUntrackedParameter<bool>("alwaysPass", false);
  scalarsToken_ = iC.consumes<DcsStatusCollection>(std::string("scalersRawToDigi"));
  dcsRecordToken_ = iC.consumes<DCSRecord>(std::string("onlineMetaDataDigis"));

  detectorOn_ = false;
  if (verbose_)
    edm::LogPrint("JetMETDQMDCSFilter") << " constructor: " << detectorTypes_ << std::endl;

  // initialize variables
  initializeVars();
}

JetMETDQMDCSFilter::JetMETDQMDCSFilter(const std::string& detectorTypes,
                                       edm::ConsumesCollector& iC,
                                       const bool verbose,
                                       const bool alwaysPass) {
  verbose_ = verbose;
  detectorTypes_ = detectorTypes;
  filter_ = !alwaysPass;
  scalarsToken_ = iC.consumes<DcsStatusCollection>(std::string("scalersRawToDigi"));
  dcsRecordToken_ = iC.consumes<DCSRecord>(std::string("onlineMetaDataDigis"));

  detectorOn_ = false;
  if (verbose_)
    edm::LogPrint("JetMETDQMDCSFilter") << " constructor: " << detectorTypes_ << std::endl;

  // initialize variables
  initializeVars();
}

// initialize
void JetMETDQMDCSFilter::initializeVars() {
  passPIX = false, passSiStrip = false;
  passECAL = false, passES = false;
  passHBHE = false, passHF = false, passHO = false;
  passMuon = false;

  passPerDet_ = {{"pixel", false},
                 {"sistrip", false},
                 {"ecal", false},
                 {"hbhe", false},
                 {"hf", false},
                 {"ho", false},
                 {"es", false},
                 {"muon", false}};
}

//
// -- Destructor
//
JetMETDQMDCSFilter::~JetMETDQMDCSFilter() {
  if (verbose_)
    edm::LogPrint("JetMETDQMDCSFilter") << " destructor: " << std::endl;
}

template <typename T>
void JetMETDQMDCSFilter::checkDCSInfoPerPartition(const T& DCS) {
  if (associationMap_.empty()) {
    associationMap_ = {{"pixel", {T::BPIX, T::FPIX}},
                       {"sistrip", {T::TIBTID, T::TOB, T::TECp, T::TECm}},
                       {"ecal", {T::EBp, T::EBm, T::EEp, T::EEm}},
                       {"hbhe", {T::HBHEa, T::HBHEb, T::HBHEc}},
                       {"hf", {T::HF}},
                       {"ho", {T::HO}},
                       {"es", {T::ESp, T::ESm}},
                       {"muon", {T::RPC, T::DT0, T::DTp, T::DTm, T::CSCp, T::CSCm}}};
  }

  for (const auto& [detName, listOfParts] : associationMap_) {
    if (detectorTypes_.find(detName) != std::string::npos) {
      bool ANDofParts{true};
      for (const auto& part : listOfParts) {
        if constexpr (std::is_same_v<T, DcsStatus>) {
          ANDofParts &= DCS.ready(part);
        } else if constexpr (std::is_same_v<T, DCSRecord>) {
          ANDofParts &= DCS.highVoltageReady(part);
        } else {
          edm::LogError("JetMETDQMDCSFilter")
              << __FILE__ << " " << __LINE__ << " passed a wrong object type, cannot deduce DCS information.\n"
              << " returning true" << std::endl;
          ANDofParts &= true;
        }
      }

      if (ANDofParts) {
        if (verbose_) {
          edm::LogPrint("JetMETDQMDCSFilter") << detName << " on" << std::endl;
        }
        passPerDet_[detName] = true;
      } else {
        detectorOn_ = false;
      }
    }  // if it matches the requested detname
  }    // loop on partitions
}

bool JetMETDQMDCSFilter::filter(const edm::Event& evt, const edm::EventSetup& es) {
  detectorOn_ = true;

  if (!evt.isRealData())
    return detectorOn_;
  if (!filter_)
    return detectorOn_;

  edm::Handle<DcsStatusCollection> dcsStatus;
  evt.getByToken(scalarsToken_, dcsStatus);

  edm::Handle<DCSRecord> dcsRecord;
  evt.getByToken(dcsRecordToken_, dcsRecord);

  if (dcsStatus.isValid() && !dcsStatus->empty()) {
    checkDCSInfoPerPartition((*dcsStatus)[0]);
  } else if (dcsRecord.isValid()) {
    checkDCSInfoPerPartition(*dcsRecord);
  } else {
    edm::LogError("JetMETDQMDCSFilter")
        << "Error! can't get the product, neither DCSRecord, nor scalersRawToDigi: accept in any case!";
  }

  // assign the values
  passPIX = passPerDet_["pixel"];
  passSiStrip = passPerDet_["sistrip"];
  passECAL = passPerDet_["ecal"];
  passES = passPerDet_["es"];
  passHBHE = passPerDet_["hbhe"];
  passHF = passPerDet_["hf"];
  passHO = passPerDet_["ho"];
  passMuon = passPerDet_["muon"];

  if (verbose_) {
    if (detectorOn_)
      edm::LogPrint("JetMETDQMDCSFilter") << "event pass!";

    std::stringstream ss;
    for (const auto& [detName, result] : passPerDet_) {
      if (detectorTypes_.find(detName) != std::string::npos) {
        ss << "Passes " << detName << ": " << (result ? "True\n" : "False\n");
      }
    }

    edm::LogPrint("JetMETDQMDCSFilter") << ss.str();
  }

  return detectorOn_;
}
