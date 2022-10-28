// -*- C++ -*-
//
// Package:    HcalTasks
// Class:      HcalFEDIntegrityTask
// Original Author: Long Wang - University of Maryland
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

#include <vector>
#include <string>

using namespace std;
using namespace edm;

class HcalFEDIntegrityTask : public DQMEDAnalyzer {
public:
  HcalFEDIntegrityTask(const edm::ParameterSet &ps);
  ~HcalFEDIntegrityTask() override;

  void dqmBeginRun(const edm::Run &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void labelBins(MonitorElement *me);

  edm::EDGetTokenT<FEDRawDataCollection> tokFEDs_;
  edm::EDGetTokenT<HcalUnpackerReport> tokReport_;

  int numOfFED_, minFEDNum_, maxFEDNum_;
  std::string dirName_;
  MonitorElement *meFEDEntries_;
  MonitorElement *meFEDFatal_;
  MonitorElement *meFEDNonFatal_;
};

HcalFEDIntegrityTask::HcalFEDIntegrityTask(const edm::ParameterSet &ps)
    : tokFEDs_(consumes<FEDRawDataCollection>(
          ps.getUntrackedParameter<edm::InputTag>("tagFEDs", edm::InputTag("rawDataCollector")))),
      tokReport_(consumes<HcalUnpackerReport>(
          ps.getUntrackedParameter<edm::InputTag>("tagReport", edm::InputTag("hcalDigis")))),
      minFEDNum_(ps.getUntrackedParameter<int>("MinHcalFEDID", FEDNumbering::MINHCALuTCAFEDID)),
      maxFEDNum_(ps.getUntrackedParameter<int>("MaxHcalFEDID", FEDNumbering::MAXHCALuTCAFEDID)),
      dirName_(ps.getUntrackedParameter<std::string>("DirName", "Hcal/FEDIntegrity/")) {
  LogInfo("HcalDQM") << "HcalFEDIntegrityTask::HcalFEDIntegrityTask: Constructor Initialization" << endl;
  numOfFED_ = maxFEDNum_ - minFEDNum_ + 1;
}

HcalFEDIntegrityTask::~HcalFEDIntegrityTask() {
  LogInfo("HcalDQM") << "HcalFEDIntegrityTask::~HcalFEDIntegrityTask: Destructor" << endl;
}

void HcalFEDIntegrityTask::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {}

void HcalFEDIntegrityTask::bookHistograms(DQMStore::IBooker &iBooker,
                                          edm::Run const &iRun,
                                          edm::EventSetup const &iSetup) {
  iBooker.cd();
  iBooker.setCurrentFolder(dirName_);

  meFEDEntries_ = iBooker.book1D("FEDEntries", "FED Entries", numOfFED_, minFEDNum_, maxFEDNum_ + 1);
  this->labelBins(meFEDEntries_);
  meFEDFatal_ = iBooker.book1D("FEDFatal", "FED Fatal Errors", numOfFED_, minFEDNum_, maxFEDNum_ + 1);
  this->labelBins(meFEDFatal_);
  meFEDNonFatal_ = iBooker.book1D("FEDNonFatal", "FED NON Fatal Errors", numOfFED_, minFEDNum_, maxFEDNum_ + 1);
  this->labelBins(meFEDNonFatal_);
}

void HcalFEDIntegrityTask::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<FEDRawDataCollection> raw;
  edm::Handle<HcalUnpackerReport> report;
  iEvent.getByToken(tokFEDs_, raw);
  iEvent.getByToken(tokReport_, report);

  // FEDs with unpacking errors: https://github.com/cms-sw/cmssw/blob/master/EventFilter/HcalRawToDigi/plugins/HcalRawToDigi.cc#L235-L262
  const std::vector<int> FedsError = (*report).getFedsError();
  for (auto &fed : FedsError) {
    if (fed < 1000)
      LogWarning("HcalDQM") << "HcalFEDIntegrityTask::analyze: obsoleteVME FEDs from HcalUnpackerReport" << endl;
    meFEDFatal_->Fill(fed);
  }

  ///////////////////////////////////////////
  // Same checks as the RawTask Summary map
  ///////////////////////////////////////////
  for (int fed = FEDNumbering::MINHCALuTCAFEDID; fed <= FEDNumbering::MAXHCALuTCAFEDID; fed++) {
    const FEDRawData &fedRawData = raw->FEDData(fed);
    if (fedRawData.size() != 0) {
      meFEDEntries_->Fill(fed);
    }

    hcal::AMC13Header const *amc13 = (hcal::AMC13Header const *)fedRawData.data();
    if (!amc13) {
      continue;
    }

    uint32_t bcn = amc13->bunchId();
    uint32_t orn = amc13->orbitNumber() & 0xFFFF;  // LS 16bits only
    uint32_t evn = amc13->l1aNumber();
    int namc = amc13->NAMC();

    // looping over AMC in this packet
    for (int iamc = 0; iamc < namc; iamc++) {
      if (!amc13->AMCEnabled(iamc) || !amc13->AMCDataPresent(iamc) || !amc13->AMCCRCOk(iamc) ||
          amc13->AMCSegmented(iamc)) {
        LogWarning("HcalDQM") << "HcalFEDIntegrityTask::analyze: AMC issue on iamc" << iamc << endl;
        continue;
      }

      HcalUHTRData uhtr(amc13->AMCPayload(iamc), amc13->AMCSize(iamc));
      uint32_t uhtr_evn = uhtr.l1ANumber();
      uint32_t uhtr_bcn = uhtr.bunchNumber();
      uint32_t uhtr_orn = uhtr.orbitNumber();

      if (uhtr_evn != evn || uhtr_bcn != bcn || uhtr_orn != orn) {
        if (std::find(FedsError.begin(), FedsError.end(), fed) ==
            FedsError.end())  // FED not already in the error list from unpacker report
          meFEDFatal_->Fill(fed);
        break;  // one mismatch is sufficient enough to determine it's a bad data
      }
    }

  }  // end of Hcal FED looping
}

void HcalFEDIntegrityTask::labelBins(MonitorElement *me) {
  int xbins = me->getNbinsX();

  if (xbins != numOfFED_)
    return;

  for (int i = 0; i < xbins; i++) {
    const std::string xLabel = fmt::format("{}", minFEDNum_ + i);
    me->setBinLabel(i + 1, xLabel, 1);
  }
}

void HcalFEDIntegrityTask::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("name", "HcalFEDIntegrityTask");
  desc.addUntracked<int>("debug", 0);
  desc.addUntracked<edm::InputTag>("tagFEDs", edm::InputTag("rawDataCollector"));
  desc.addUntracked<edm::InputTag>("tagReport", edm::InputTag("hcalDigis"));
  desc.addUntracked<int>(
      "MinHcalFEDID",
      FEDNumbering::MINHCALuTCAFEDID);  // Assuming no more VME FEDs after LS2, according to Hcal Phase1 upgrade.
  desc.addUntracked<int>("MaxHcalFEDID", FEDNumbering::MAXHCALuTCAFEDID);
  desc.addUntracked<std::string>("DirName", "Hcal/FEDIntegrity/");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HcalFEDIntegrityTask);
