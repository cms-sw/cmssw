#include "DQM/L1TMonitorClient/interface/L1TStage2CaloLayer2DEClientSummary.h"

L1TStage2CaloLayer2DEClientSummary::L1TStage2CaloLayer2DEClientSummary(
  const edm::ParameterSet& ps):
  monitor_dir_(ps.getUntrackedParameter<std::string>("monitorDir","")),
  hlSummary(nullptr),
  jetSummary(nullptr),
  egSummary(nullptr),
  tauSummary(nullptr),
  sumSummary(nullptr)
{}

L1TStage2CaloLayer2DEClientSummary::~L1TStage2CaloLayer2DEClientSummary(){}

void L1TStage2CaloLayer2DEClientSummary::dqmEndLuminosityBlock(
  DQMStore::IBooker &ibooker,
  DQMStore::IGetter &igetter,
  const edm::LuminosityBlock& lumiSeg,
  const edm::EventSetup& c) {

  book(ibooker);
  processHistograms(igetter);
}

void L1TStage2CaloLayer2DEClientSummary::dqmEndJob(
  DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {

  book(ibooker);
  processHistograms(igetter);
}

void L1TStage2CaloLayer2DEClientSummary::book(DQMStore::IBooker &ibooker) {

  ibooker.setCurrentFolder(monitor_dir_);
  if (hlSummary == nullptr) {
    hlSummary = ibooker.book1D(
      "High level summary", "Event by event comparison summary", 5, 1, 6);
    hlSummary->setBinLabel(1, "good events");
    hlSummary->setBinLabel(2, "good jets");
    hlSummary->setBinLabel(3, "good e/gs");
    hlSummary->setBinLabel(4, "good taus");
    hlSummary->setBinLabel(5, "good sums");
  } else {
    hlSummary->Reset();
  }

  if (jetSummary == nullptr) {
    jetSummary = ibooker.book1D(
      "Jet Agreement Summary", "Jet Agreement Summary", 3, 1, 4);
    jetSummary->setBinLabel(1, "good jets");
    jetSummary->setBinLabel(2, "jets pos off only");
    jetSummary->setBinLabel(3, "jets Et off only ");
  } else {
    jetSummary->Reset();
  }

  if (egSummary == nullptr) {
    egSummary = ibooker.book1D(
      "EG Agreement Summary", "EG Agreement Summary", 6, 1, 7);
    egSummary->setBinLabel(1, "good non-iso e/gs");
    egSummary->setBinLabel(2, "non-iso e/gs pos off");
    egSummary->setBinLabel(3, "non-iso e/gs Et off");
    egSummary->setBinLabel(4, "good iso e/gs");
    egSummary->setBinLabel(5, "iso e/gs pos off");
    egSummary->setBinLabel(6, "iso e/gs Et off");
  } else {
    egSummary->Reset();
  }

  if (tauSummary == nullptr) {
    tauSummary = ibooker.book1D(
      "Tau Agreement Summary", "Tau Agremeent Summary", 6, 1, 7);
    tauSummary->setBinLabel(1, "good non-iso taus");
    tauSummary->setBinLabel(2, "non-iso taus pos off");
    tauSummary->setBinLabel(3, "non-iso taus Et off");
    tauSummary->setBinLabel(4, "good iso taus");
    tauSummary->setBinLabel(5, "iso taus pos off");
    tauSummary->setBinLabel(6, "iso taus Et off");
  } else {
    tauSummary->Reset();
  }

  if (sumSummary == nullptr) {
    sumSummary = ibooker.book1D(
      "Energy Sum Agreement Summary", "Sum Agreement Summary", 9, 1, 10);
    sumSummary->setBinLabel(1, "good sums");
    sumSummary->setBinLabel(2, "good ETT sums");
    sumSummary->setBinLabel(3, "good HTT sums");
    sumSummary->setBinLabel(4, "good MET sums");
    sumSummary->setBinLabel(5, "good MHT sums");
    sumSummary->setBinLabel(6, "good MBHF sums");
    sumSummary->setBinLabel(7, "good TowCount sums");
    sumSummary->setBinLabel(8, "good AsymCount sums");
    sumSummary->setBinLabel(9, "good CentrCount sums");
  } else {
    sumSummary->Reset();
  }

}

void L1TStage2CaloLayer2DEClientSummary::processHistograms(DQMStore::IGetter &igetter){

  // get reference to relevant summary MonitorElement instances
  // - high level summary
  // - eg agreement summary
  // - energy sum agreement summary
  // - jet agreement summary
  // - tau agreement summary

  // TH1F * hist;
  // TH1F * newHist;

  MonitorElement * hlSummary_ = igetter.get(
    monitor_dir_+"/expert/CaloL2 Object Agreement Summary");
  MonitorElement * jetSummary_ = igetter.get(
    monitor_dir_+"/expert/Jet Agreement Summary");
  MonitorElement * egSummary_ = igetter.get(
    monitor_dir_+"/expert/EG Agreement Summary");
  MonitorElement * tauSummary_ = igetter.get(
    monitor_dir_+"/expert/Tau Agreement Summary");
  MonitorElement * sumSummary_ = igetter.get(
    monitor_dir_+"/expert/Energy Sum Agreement Summary");

  // check for existance of object
  if (hlSummary_) {

    // reference the histogram in MonitorElement
    // hist = hlSummary_->getTH1F();
    // newHist = hlSummary->getTH1F();

    // double totalEvents = 0, goodEvents = 0, totalJets = 0, goodJets = 0,
    //   totalEg = 0, goodEg = 0, totalTau = 0, goodTau = 0, totalSums = 0,
    //   goodSums = 0;

    // by default show 0% agreement (for edge case when no objects are found)
    double evtRatio = 0, jetRatio = 0, egRatio = 0, tauRatio = 0, sumRatio = 0;

    double totalEvents = hlSummary_->getBinContent(1);
    double goodEvents  = hlSummary_->getBinContent(2);
    double totalJets   = hlSummary_->getBinContent(3);
    double goodJets    = hlSummary_->getBinContent(4);
    double totalEg     = hlSummary_->getBinContent(5);
    double goodEg      = hlSummary_->getBinContent(6);
    double totalTau    = hlSummary_->getBinContent(7);
    double goodTau     = hlSummary_->getBinContent(8);
    double totalSums   = hlSummary_->getBinContent(9);
    double goodSums    = hlSummary_->getBinContent(10);

    if (totalEvents != 0)
      evtRatio = goodEvents / totalEvents;

    if (totalJets != 0)
      jetRatio = goodJets / totalJets;

    if (totalEg != 0)
      egRatio  = goodEg / totalEg;

    if (totalTau != 0)
      tauRatio = goodTau / totalTau;

    if (totalSums != 0)
      sumRatio = goodSums / totalSums;

    hlSummary->setBinContent(1, evtRatio);
    hlSummary->setBinContent(2, jetRatio);
    hlSummary->setBinContent(3, egRatio);
    hlSummary->setBinContent(4, tauRatio);
    hlSummary->setBinContent(5, sumRatio);
  }

  if (jetSummary_) {

    // double totalJets = 0, goodJets = 0, jetPosOff = 0, jetEtOff = 0;

    // by default show 0% agreement (for edge case when no objects are found)
    double goodRatio = 0, posOffRatio = 0, etOffRatio = 0;

    // hist = jetSummary_->getTH1F();
    // newHist = jetSummary->getTH1F();

    double totalJets = jetSummary_->getBinContent(1);
    double goodJets  = jetSummary_->getBinContent(2);
    double jetPosOff = jetSummary_->getBinContent(3);
    double jetEtOff  = jetSummary_->getBinContent(4);

    if (totalJets != 0) {
      goodRatio = goodJets / totalJets;
      posOffRatio = jetPosOff / totalJets;
      etOffRatio = jetEtOff / totalJets;
    }

    jetSummary->setBinContent(1, goodRatio);
    jetSummary->setBinContent(2, posOffRatio);
    jetSummary->setBinContent(3, etOffRatio);
  }

  if (egSummary_) {

    // double totalEgs = 0, goodEgs = 0, egPosOff = 0, egEtOff = 0,
    //   totalIsoEgs = 0, goodIsoEgs = 0, isoEgPosOff = 0, isoEgEtOff = 0;

    // by default show 0% agreement (for edge case when no objects are found)
    double goodEgRatio = 0, egPosOffRatio = 0, egEtOffRatio = 0,
       goodIsoEgRatio = 0, isoEgPosOffRatio = 0, isoEgEtOffRatio = 0;

    // hist = egSummary_->getTH1F();
    // newHist = egSummary->getTH1F();

    double totalEgs = egSummary_->getBinContent(1);
    double goodEgs  = egSummary_->getBinContent(2);
    double egPosOff = egSummary_->getBinContent(3);
    double egEtOff  = egSummary_->getBinContent(4);

    double totalIsoEgs = egSummary_->getBinContent(5);
    double goodIsoEgs  = egSummary_->getBinContent(6);
    double isoEgPosOff = egSummary_->getBinContent(7);
    double isoEgEtOff  = egSummary_->getBinContent(8);

    if (totalEgs != 0) {
      goodEgRatio = goodEgs / totalEgs;
      egPosOffRatio = egPosOff / totalEgs;
      egEtOffRatio = egEtOff / totalEgs;
    }

    if (totalIsoEgs != 0) {
      goodIsoEgRatio = goodIsoEgs / totalIsoEgs;
      isoEgPosOffRatio = isoEgPosOff / totalIsoEgs;
      isoEgEtOffRatio = isoEgEtOff / totalIsoEgs;
    }

    egSummary->setBinContent(1, goodEgRatio);
    egSummary->setBinContent(2, egPosOffRatio);
    egSummary->setBinContent(3, egEtOffRatio);

    egSummary->setBinContent(4, goodIsoEgRatio);
    egSummary->setBinContent(5, isoEgPosOffRatio);
    egSummary->setBinContent(6, isoEgEtOffRatio);
  }

  if (tauSummary_) {

    // double totalTaus = 0, goodTaus = 0, tauPosOff = 0, tauEtOff = 0,
    //   totalIsoTaus = 0, goodIsoTaus = 0, isoTauPosOff = 0, isoTauEtOff = 0;

    // by default show 0% agreement (for edge case when no objects are found)
    double goodTauRatio = 0, tauPosOffRatio = 0, tauEtOffRatio = 0,
      goodIsoTauRatio = 0, isoTauPosOffRatio= 0, isoTauEtOffRatio = 0;

    // hist = tauSummary_->getTH1F();
    // newHist = tauSummary->getTH1F();

    double totalTaus = tauSummary_->getBinContent(1);
    double goodTaus  = tauSummary_->getBinContent(2);
    double tauPosOff = tauSummary_->getBinContent(3);
    double tauEtOff  = tauSummary_->getBinContent(4);

    double totalIsoTaus = tauSummary_->getBinContent(5);
    double goodIsoTaus  = tauSummary_->getBinContent(6);
    double isoTauPosOff = tauSummary_->getBinContent(7);
    double isoTauEtOff  = tauSummary_->getBinContent(8);

    if (totalTaus != 0) {
      goodTauRatio = goodTaus / totalTaus;
      tauPosOffRatio = tauPosOff / totalTaus;
      tauEtOffRatio = tauEtOff / totalTaus;
    }

    if (totalIsoTaus != 0) {
      goodIsoTauRatio = goodIsoTaus / totalIsoTaus;
      isoTauPosOffRatio = isoTauPosOff / totalIsoTaus;
      isoTauEtOffRatio = isoTauEtOff / totalIsoTaus;
    }

    tauSummary->setBinContent(1, goodTauRatio);
    tauSummary->setBinContent(2, tauPosOffRatio);
    tauSummary->setBinContent(3, tauEtOffRatio);

    tauSummary->setBinContent(4, goodIsoTauRatio);
    tauSummary->setBinContent(5, isoTauPosOffRatio);
    tauSummary->setBinContent(6, isoTauEtOffRatio);
  }

  if (sumSummary_) {

    // double totalSums = 0, goodSums = 0, totalETT = 0, goodETT = 0, totalHTT = 0,
    //   goodHTT = 0, totalMET = 0, goodMET = 0, totalMHT = 0, goodMHT = 0,
    //   totalMBHF = 0, goodMBHF = 0, totalTowCount = 0, goodTowCount = 0

    // by default show 0% agreement (for edge case when no objects are found)
    double goodSumRatio = 0, goodETTRatio = 0, goodHTTRatio = 0,
      goodMETRatio = 0, goodMHTRatio = 0, goodMBHFRatio = 0,
      goodTowCountRatio = 0, goodAsymCountRatio = 0, goodCentrCountRatio = 0; 

    double totalSums     = sumSummary_->getBinContent(1);
    double goodSums      = sumSummary_->getBinContent(2);
    double totalETT      = sumSummary_->getBinContent(3);
    double goodETT       = sumSummary_->getBinContent(4);
    double totalHTT      = sumSummary_->getBinContent(5);
    double goodHTT       = sumSummary_->getBinContent(6);
    double totalMET      = sumSummary_->getBinContent(7);
    double goodMET       = sumSummary_->getBinContent(8);
    double totalMHT      = sumSummary_->getBinContent(9);
    double goodMHT       = sumSummary_->getBinContent(10);
    double totalMBHF     = sumSummary_->getBinContent(11);
    double goodMBHF      = sumSummary_->getBinContent(12);
    double totalTowCount = sumSummary_->getBinContent(13);
    double goodTowCount  = sumSummary_->getBinContent(14);
    double totalAsymCount= sumSummary_->getBinContent(15);
    double goodAsymCount = sumSummary_->getBinContent(16);
    double totalCentrCount= sumSummary_->getBinContent(17);
    double goodCentrCount = sumSummary_->getBinContent(18);
    if (totalSums)
      goodSumRatio = goodSums / totalSums;

    if (totalETT)
      goodETTRatio = goodETT / totalETT;

    if (totalHTT)
      goodHTTRatio = goodHTT / totalHTT;

    if (totalMET)
      goodMETRatio = goodMET / totalMET;

    if (totalMHT)
      goodMHTRatio = goodMHT / totalMHT;

    if (totalMBHF)
      goodMBHFRatio = goodMBHF / totalMBHF;

    if (totalTowCount)
      goodTowCountRatio = goodTowCount / totalTowCount;
        
    if (totalAsymCount)
      goodAsymCountRatio = goodAsymCount / totalAsymCount;

    if (totalCentrCount)
      goodCentrCountRatio = goodCentrCount / totalCentrCount;

    sumSummary->setBinContent(1, goodSumRatio);
    sumSummary->setBinContent(2, goodETTRatio);
    sumSummary->setBinContent(3, goodHTTRatio);
    sumSummary->setBinContent(4, goodMETRatio);
    sumSummary->setBinContent(5, goodMHTRatio);
    sumSummary->setBinContent(6, goodMBHFRatio);
    sumSummary->setBinContent(7, goodTowCountRatio);
    sumSummary->setBinContent(8, goodAsymCountRatio);
    sumSummary->setBinContent(9, goodCentrCountRatio);
  }

  }
