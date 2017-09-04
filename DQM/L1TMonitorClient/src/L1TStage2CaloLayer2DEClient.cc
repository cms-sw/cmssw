#include "DQM/L1TMonitorClient/interface/L1TStage2CaloLayer2DEClient.h"

L1TStage2CaloLayer2DEClient::L1TStage2CaloLayer2DEClient(
  const edm::ParameterSet& ps):
  monitor_dir_(ps.getUntrackedParameter<std::string>("monitorDir","")),
  input_dir_data_(ps.getUntrackedParameter<std::string>("inputDataDir","")),
  input_dir_emul_(ps.getUntrackedParameter<std::string>("inputEmulDir",""))
{}

L1TStage2CaloLayer2DEClient::~L1TStage2CaloLayer2DEClient(){}

void L1TStage2CaloLayer2DEClient::dqmEndLuminosityBlock(
  DQMStore::IBooker &ibooker,
  DQMStore::IGetter &igetter,
  const edm::LuminosityBlock& lumiSeg,
  const edm::EventSetup& c) {

  book(ibooker);
  processHistograms(igetter);
}

void L1TStage2CaloLayer2DEClient::dqmEndJob(
  DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {

  book(ibooker);
  processHistograms(igetter);
}

void L1TStage2CaloLayer2DEClient::book(DQMStore::IBooker &ibooker) {

  ibooker.setCurrentFolder("L1TEMU/L1TdeStage2CaloLayer2");
  hlSummary = ibooker.book1D(
    "High level summary", "Event by event comparison summary", 5, 1, 6);
  hlSummary->setBinLabel(1, "good events");
  hlSummary->setBinLabel(2, "good jets");
  hlSummary->setBinLabel(3, "good e/gs");
  hlSummary->setBinLabel(4, "good taus");
  hlSummary->setBinLabel(5, "good sums");

  jetSummary = ibooker.book1D(
    "Jet Agreement Summary", "Jet Agreement Summary", 3, 1, 4);
  jetSummary->setBinLabel(1, "good jets");
  jetSummary->setBinLabel(2, "jets pos off only");
  jetSummary->setBinLabel(3, "jets Et off only ");

  egSummary = ibooker.book1D(
    "EG Agreement Summary", "EG Agreement Summary", 6, 1, 7);
  egSummary->setBinLabel(1, "good non-iso e/gs");
  egSummary->setBinLabel(2, "non-iso e/gs pos off");
  egSummary->setBinLabel(3, "non-iso e/gs Et off");
  egSummary->setBinLabel(4, "good iso e/gs");
  egSummary->setBinLabel(5, "iso e/gs pos off");
  egSummary->setBinLabel(6, "iso e/gs Et off");

  tauSummary = ibooker.book1D(
    "Tau Agreement Summary", "Tau Agremeent Summary", 6, 1, 7);
  tauSummary->setBinLabel(1, "good non-iso taus");
  tauSummary->setBinLabel(2, "non-iso taus pos off");
  tauSummary->setBinLabel(3, "non-iso taus Et off");
  tauSummary->setBinLabel(4, "good iso taus");
  tauSummary->setBinLabel(5, "iso taus pos off");
  tauSummary->setBinLabel(6, "iso taus Et off");

  sumSummary = ibooker.book1D(
    "Energy Sum Agreement Summary", "Sum Agreement Summary", 7, 1, 8);
  sumSummary->setBinLabel(1, "good sums");
  sumSummary->setBinLabel(2, "good ETT sums");
  sumSummary->setBinLabel(3, "good HTT sums");
  sumSummary->setBinLabel(4, "good MET sums");
  sumSummary->setBinLabel(5, "good MHT sums");
  sumSummary->setBinLabel(6, "good MBHF sums");
  sumSummary->setBinLabel(7, "good TowCount sums");

  // problemSummary;

  ibooker.setCurrentFolder(monitor_dir_);

  CenJetRankComp_ = ibooker.book1D(
    "CenJetsRankDERatio","Data/Emul of Central Jet E_{T}; Jet iE_{T}; Counts",
    2048, -0.5, 2047.5);
  CenJetEtaComp_ = ibooker.book1D(
    "CenJetsEtaDERatio","Data/Emul of Central Jet #eta; Jet i#eta; Counts",
    229, -114.5, 114.5);
  CenJetPhiComp_ = ibooker.book1D(
    "CenJetsPhiDERatio","Data/Emul of Central Jet #phi; Jet i#phi; Counts",
    144, -0.5, 143.5);
  ForJetRankComp_ = ibooker.book1D(
    "ForJetsRankDERatio","Data/Emul of Forward Jet E_{T}; Jet iE_{T}; Counts",
    2048, -0.5, 2047.5);
  ForJetEtaComp_ = ibooker.book1D(
    "ForJetsEtaDERatio","Data/Emul of Forward Jet #eta; Jet i#eta; Counts",
    229, -114.5, 114.5);
  ForJetPhiComp_ = ibooker.book1D(
    "ForJetsPhiDERatio","Data/Emul of Forward Jet #phi; Jet i#phi; Counts",
    144, -0.5, 143.5);
  IsoEGRankComp_ = ibooker.book1D(
    "IsoEGRankDERatio","Data/Emul of isolated eg E_{T}; EG iE_{T}; Counts",
    512, -0.5, 511.5);
  IsoEGEtaComp_ = ibooker.book1D(
    "IsoEGEtaDERatio","Data/Emul of isolated eg #eta; EG i#eta; Counts",
    229, -114.5, 114.5);
  IsoEGPhiComp_ = ibooker.book1D(
    "IsoEGPhiDERatio","Data/Emul of isolated eg #phi; EG i#eta; Counts",
    144, -0.5, 143.5);
  NonIsoEGRankComp_ = ibooker.book1D(
    "NonIsoEGRankDERatio",
    "Data/Emul of non-isolated eg E_{T}; EG iE_{T}; Counts",
    512, -0.5, 511.5);
  NonIsoEGEtaComp_ = ibooker.book1D(
    "NonIsoEGEtaDERatio","Data/Emul of non-isolated eg #eta; EG i#eta; Counts",
    229, -114.5, 114.5);
  NonIsoEGPhiComp_ = ibooker.book1D(
    "NonIsoEGPhiDERatio","Data/Emul of non-isolated eg #phi; EG i#phi; Counts",
    144, -0.5, 143.5);
  TauRankComp_ = ibooker.book1D(
    "TauRankDERatio","Data/Emul of relax tau E_{T}; Tau iE_{T}; Counts",
    512, -0.5, 511.5);
  TauEtaComp_ = ibooker.book1D(
    "TauEtaDERatio","Data/Emul of relax tau #eta; Tau i#eta; Counts",
    229, -114.5, 114.5);
  TauPhiComp_ = ibooker.book1D(
    "TauPhiDERatio","Data/Emul of relax tau eg #phi; Tau i#phi; Counts",
    144, -0.5, 143.5);
  IsoTauRankComp_ = ibooker.book1D(
    "IsoTauRankDERatio","Data/Emul of iso tau E_{T}; ISO Tau iE_{T}; Counts",
    512, -0.5, 511.5);
  IsoTauEtaComp_ = ibooker.book1D(
    "IsoTauEtaDERatio","Data/Emul of iso tau #eta; ISO Tau i#eta; Counts",
    229, -114.5, 114.5);
  IsoTauPhiComp_ = ibooker.book1D(
    "IsoTauPhiDERatio","Data/Emul of iso tau #phi; ISO Tau i#phi; Counts",
    144, -0.5, 143.5);
  METComp_ = ibooker.book1D(
    "METRatio","Data/Emul of MET; iE_{T}; Events",
    4096, -0.5, 4095.5);
  METPhiComp_ = ibooker.book1D(
    "METPhiRatio","Data/Emul of MET #phi; MET i#phi; Events",
    1008, -0.5, 1007.5);
  METHFComp_ = ibooker.book1D(
    "METHFRatio","Data/Emul of METHF; METHF iE_{T}; Events",
    4096, -0.5, 4095.5);
  METHFPhiComp_ = ibooker.book1D(
    "METHFPhiRatio","Data/Emul of METHF #phi; METHF i#phi; Events",
    1008, -0.5, 1007.5);
  MHTComp_ = ibooker.book1D(
    "MHTRatio","Data/Emul of MHT; MHT iE_{T}; Events",
    4096, -0.5, 4095.5);
  METPhiComp_ = ibooker.book1D(
    "MHTPhiRatio","Data/Emul of MHT #phi; MHTHF i#phi; Events",
    1008, -0.5, 1007.5);
  MHTHFComp_ = ibooker.book1D(
    "MHTHFRatio","Data/Emul of MHTHF; MHTHF iE_{T}; Events",
    4096, -0.5, 4095.5);
  MHTPhiComp_ = ibooker.book1D(
    "MHTHFPhiRatio","Data/Emul of MHTHF #phi; MHTHF i#phi; Events",
    1008, -0.5, 1007.5);
  ETTComp_ = ibooker.book1D(
    "ETTRatio","Data/Emul of ET Total; ETT iE_{T}; Events",
    4096, -0.5, 4095.5);
  ETTEMComp_ = ibooker.book1D(
    "ETTEMRatio","Data/Emul of ET Total EM; ETTEM iE_{T}; Events",
    4096, -0.5, 4095.5);
  HTTComp_ = ibooker.book1D(
    "HTTRatio","Data/Emul of HT Total; HT iE_{T}; Events",
    4096, -0.5, 4095.5);

  MinBiasHFP0Comp_ = ibooker.book1D(
    "MinBiasHFP0Ratio", "Data/Emul MinBiasHFP0; N_{towers}; Events",
    16, -0.5, 15.5);
  MinBiasHFM0Comp_ = ibooker.book1D(
    "MinBiasHFM0Ratio", "Data/Emul MinBiasHFM0; N_{towers}; Events",
    16, -0.5, 15.5);
  MinBiasHFP1Comp_ = ibooker.book1D(
    "MinBiasHFP1Ratio", "Data/Emul MinBiasHFP1; N_{towers}; Events",
    16, -0.5, 15.5);
  MinBiasHFM1Comp_ = ibooker.book1D(
    "MinBiasHFM1Ratio", "Data/Emul MinBiasHFM1; N_{towers}; Events",
    16, -0.5, 15.5);

  TowerCountComp_ = ibooker.book1D(
    "TowCountRatio", "Data/Emul Tower Count; N_{towers}; Events",
    5904, -0.5, 5903.5);
}

void L1TStage2CaloLayer2DEClient::processHistograms(DQMStore::IGetter &igetter){

  // get reference to relevant summary MonitorElement instances
  // - high level summary
  // - eg agreement summary
  // - energy sum agreement summary
  // - jet agreement summary
  // - tau agreement summary

  // TH1F * hist;
  // TH1F * newHist;

  MonitorElement * hlSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/expert/CaloL2 Object Agreement Summary");
  MonitorElement * jetSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/expert/Jet Agreement Summary");
  MonitorElement * egSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/expert/EG Agreement Summary");
  MonitorElement * tauSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/expert/Tau Agreement Summary");
  MonitorElement * sumSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/expert/Energy Sum Agreement Summary");

  // check for existance of object
  if (hlSummary_) {

    // reference the histogram in MonitorElement
    // hist = hlSummary_->getTH1F();
    // newHist = hlSummary->getTH1F();

    // double totalEvents = 0, goodEvents = 0, totalJets = 0, goodJets = 0,
    //   totalEg = 0, goodEg = 0, totalTau = 0, goodTau = 0, totalSums = 0,
    //   goodSums = 0;

    // by default show 100% agreement (for edge case when no objects are found)
    double evtRatio = 1, jetRatio = 1, egRatio = 1, tauRatio = 1, sumRatio = 1;

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

    // by default show 100% agreement (for edge case when no objects are found)
    double goodRatio = 1, posOffRatio = 0, etOffRatio = 0;

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

    // by default show 100% agreement (for edge case when no objects are found)
    double goodEgRatio = 1, egPosOffRatio = 0, egEtOffRatio = 0,
       goodIsoEgRatio = 1, isoEgPosOffRatio = 0, isoEgEtOffRatio = 0;

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

    // by default show 100% agreement (for edge case when no objects are found)
    double goodTauRatio = 1, tauPosOffRatio = 0, tauEtOffRatio = 0,
      goodIsoTauRatio = 1, isoTauPosOffRatio= 0, isoTauEtOffRatio = 0;

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

    // by default show 100% agreement (for edge case when no objects are found)
    double goodSumRatio = 1, goodETTRatio = 1, goodHTTRatio = 1,
      goodMETRatio = 1, goodMHTRatio = 1, goodMBHFRatio = 1,
      goodTowCountRatio = 1;

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

    sumSummary->setBinContent(1, goodSumRatio);
    sumSummary->setBinContent(2, goodETTRatio);
    sumSummary->setBinContent(3, goodHTTRatio);
    sumSummary->setBinContent(4, goodMETRatio);
    sumSummary->setBinContent(5, goodMHTRatio);
    sumSummary->setBinContent(6, goodMBHFRatio);
    sumSummary->setBinContent(7, goodTowCountRatio);
  }

  MonitorElement* dataHist_;
  MonitorElement* emulHist_;

  // central jets
  dataHist_ = igetter.get(input_dir_data_+"/Central-Jets/"+"CenJetsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Central-Jets/"+"CenJetsRank");

  if (dataHist_ && emulHist_){
    TH1F *cjrNum = dataHist_->getTH1F();
    TH1F *cjrDen = emulHist_->getTH1F();

    TH1F *CenJetRankRatio = CenJetRankComp_->getTH1F();

    CenJetRankRatio->Divide(cjrNum, cjrDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Central-Jets/"+"CenJetsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/Central-Jets/"+"CenJetsEta");

  if (dataHist_ && emulHist_){
    TH1F *cjeNum = dataHist_->getTH1F();
    TH1F *cjeDen = emulHist_->getTH1F();

    TH1F *CenJetEtaRatio = CenJetEtaComp_->getTH1F();

    CenJetEtaRatio->Divide(cjeNum, cjeDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Central-Jets/"+"CenJetsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/Central-Jets/"+"CenJetsPhi");

  if (dataHist_ && emulHist_){
    TH1F *cjpNum = dataHist_->getTH1F();
    TH1F *cjpDen = emulHist_->getTH1F();

    TH1F *CenJetPhiRatio = CenJetPhiComp_->getTH1F();

    CenJetPhiRatio->Divide(cjpNum, cjpDen);

  }

  // forward jets
  dataHist_ = igetter.get(input_dir_data_+"/Forward-Jets/"+"ForJetsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Forward-Jets/"+"ForJetsRank");

  if (dataHist_ && emulHist_){

    TH1F *fjrNum = dataHist_->getTH1F();
    TH1F *fjrDen = emulHist_->getTH1F();

    TH1F *ForJetRankRatio = ForJetRankComp_->getTH1F();

    ForJetRankRatio->Divide(fjrNum, fjrDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Forward-Jets/"+"ForJetsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/Forward-Jets/"+"ForJetsEta");

  if (dataHist_ && emulHist_){
    TH1F *fjeNum = dataHist_->getTH1F();
    TH1F *fjeDen = emulHist_->getTH1F();

    TH1F *ForJetEtaRatio = ForJetEtaComp_->getTH1F();

    ForJetEtaRatio->Divide(fjeNum, fjeDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Forward-Jets/"+"ForJetsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/Forward-Jets/"+"ForJetsPhi");

  if (dataHist_ && emulHist_){
    TH1F *fjpNum = dataHist_->getTH1F();
    TH1F *fjpDen = emulHist_->getTH1F();

    TH1F *ForJetPhiRatio = ForJetPhiComp_->getTH1F();

    ForJetPhiRatio->Divide(fjpNum, fjpDen);
  }

  // isolated eg

  dataHist_ = igetter.get(input_dir_data_+"/Isolated-EG/"+"IsoEGsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Isolated-EG/"+"IsoEGsRank");

  if (dataHist_ && emulHist_){
    TH1F *ierNum = dataHist_->getTH1F();
    TH1F *ierDen = emulHist_->getTH1F();

    TH1F *IsoEGRankRatio = IsoEGRankComp_->getTH1F();

    IsoEGRankRatio->Divide(ierNum, ierDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Isolated-EG/"+"IsoEGsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/Isolated-EG/"+"IsoEGsEta");

  if (dataHist_ && emulHist_){
    TH1F *ieeNum = dataHist_->getTH1F();
    TH1F *ieeDen = emulHist_->getTH1F();

    TH1F *IsoEGEtaRatio = IsoEGEtaComp_->getTH1F();

    IsoEGEtaRatio->Divide(ieeNum, ieeDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Isolated-EG/"+"IsoEGsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/Isolated-EG/"+"IsoEGsPhi");

  if (dataHist_ && emulHist_){
    TH1F *iepNum = dataHist_->getTH1F();
    TH1F *iepDen = emulHist_->getTH1F();

    TH1F *IsoEGPhiRatio = IsoEGPhiComp_->getTH1F();

    IsoEGPhiRatio->Divide(iepNum, iepDen);
  }

  // non-isolated eg
  dataHist_ = igetter.get(input_dir_data_+"/NonIsolated-EG/"+"NonIsoEGsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/NonIsolated-EG/"+"NonIsoEGsRank");

  if (dataHist_ && emulHist_){
    TH1F *nerNum = dataHist_->getTH1F();
    TH1F *nerDen = emulHist_->getTH1F();

    TH1F *NonIsoEGRankRatio = NonIsoEGRankComp_->getTH1F();

    NonIsoEGRankRatio->Divide(nerNum, nerDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/NonIsolated-EG/"+"NonIsoEGsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/NonIsolated-EG/"+"NonIsoEGsEta");

  if (dataHist_ && emulHist_){
    TH1F *neeNum = dataHist_->getTH1F();
    TH1F *neeDen = emulHist_->getTH1F();

    TH1F *NonIsoEGEtaRatio = NonIsoEGEtaComp_->getTH1F();

    NonIsoEGEtaRatio->Divide(neeNum, neeDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/NonIsolated-EG/"+"NonIsoEGsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/NonIsolated-EG/"+"NonIsoEGsPhi");

  if (dataHist_ && emulHist_){
    TH1F *nepNum = dataHist_->getTH1F();
    TH1F *nepDen = emulHist_->getTH1F();

    TH1F *NonIsoEGPhiRatio = NonIsoEGPhiComp_->getTH1F();

    NonIsoEGPhiRatio->Divide(nepNum, nepDen);
  }

  // rlx tau
  dataHist_ = igetter.get(input_dir_data_+"/NonIsolated-Tau/"+"TausRank");
  emulHist_ = igetter.get(input_dir_emul_+"/NonIsolated-Tau/"+"TausRank");

  if (dataHist_ && emulHist_){
    TH1F *trNum = dataHist_->getTH1F();
    TH1F *trDen = emulHist_->getTH1F();

    TH1F *TauRankRatio = TauRankComp_->getTH1F();

    TauRankRatio->Divide(trNum, trDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/NonIsolated-Tau/"+"TausEta");
  emulHist_ = igetter.get(input_dir_emul_+"/NonIsolated-Tau/"+"TausEta");

  if (dataHist_ && emulHist_){
    TH1F *teNum = dataHist_->getTH1F();
    TH1F *teDen = emulHist_->getTH1F();

    TH1F *TauEtaRatio = TauEtaComp_->getTH1F();

    TauEtaRatio->Divide(teNum, teDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/NonIsolated-Tau/"+"TausPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/NonIsolated-Tau/"+"TausPhi");

  if (dataHist_ && emulHist_){
    TH1F *tpNum = dataHist_->getTH1F();
    TH1F *tpDen = emulHist_->getTH1F();

    TH1F *TauPhiRatio = TauPhiComp_->getTH1F();

    TauPhiRatio->Divide(tpNum, tpDen);
  }

  // iso tau
  dataHist_ = igetter.get(input_dir_data_+"/Isolated-Tau/"+"IsoTausRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Isolated-Tau/"+"IsoTausRank");

  if (dataHist_ && emulHist_){
    TH1F *itrNum = dataHist_->getTH1F();
    TH1F *itrDen = emulHist_->getTH1F();

    TH1F *IsoTauRankRatio = IsoTauRankComp_->getTH1F();

    IsoTauRankRatio->Divide(itrNum, itrDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Isolated-Tau/"+"IsoTausEta");
  emulHist_ = igetter.get(input_dir_emul_+"/Isolated-Tau/"+"IsoTausEta");

  if (dataHist_ && emulHist_){
    TH1F *iteNum = dataHist_->getTH1F();
    TH1F *iteDen = emulHist_->getTH1F();

    TH1F *IsoTauEtaRatio = IsoTauEtaComp_->getTH1F();

    IsoTauEtaRatio->Divide(iteNum, iteDen);
  }

  dataHist_ = igetter.get(input_dir_data_+"/Isolated-Tau/"+"IsoTausPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/Isolated-Tau/"+"IsoTausPhi");

  if (dataHist_ && emulHist_){
    TH1F *itpNum = dataHist_->getTH1F();
    TH1F *itpDen = emulHist_->getTH1F();

    TH1F *IsoTauPhiRatio = IsoTauPhiComp_->getTH1F();

    IsoTauPhiRatio->Divide(itpNum, itpDen);
  }

  // MET
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"METRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"METRank");

  if (dataHist_ && emulHist_){
    TH1F *metNum = dataHist_->getTH1F();
    TH1F *metDen = emulHist_->getTH1F();

    TH1F *METRatio = METComp_->getTH1F();

    METRatio->Divide(metNum, metDen);
  }

  // This causes CMSSW to segfault with a complaint that ROOT cannot divide two
  // histograms with different number of bins. Checking the contents of the data
  // and emulator histograms, using both GetNbinsX() and GetSize(), it can be
  // seen that the reported number of bins is the same. This needs more
  // investigation.

  // MET Phi
  // dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"METPhi");
  // emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"METPhi");

  // if (dataHist_ && emulHist_){
  //   TH1F * metphiNum = dataHist_->getTH1F();
  //   TH1F * metphiDen = emulHist_->getTH1F();

  //   TH1F * METPhiRatio = METPhiComp_->getTH1F();

  //   std::cout << "data met " << metphiNum->GetNbinsX() << std::endl;
  //   std::cout << "emul met " << metphiDen->GetNbinsX() << std::endl;

  //   METPhiRatio->Divide(metphiNum, metphiDen);
  // }

  // METHF
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"METHFRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"METHFRank");

  if (dataHist_ && emulHist_){
    TH1F *methfNum = dataHist_->getTH1F();
    TH1F *methfDen = emulHist_->getTH1F();

    TH1F *METHFRatio = METHFComp_->getTH1F();

    METHFRatio->Divide(methfNum, methfDen);
  }

  // // METHF Phi
  // dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"METHFPhi");
  // emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"METHFPhi");

  // if (dataHist_ && emulHist_){
  //   TH1F *methfphiNum = dataHist_->getTH1F();
  //   TH1F *methfphiDen = emulHist_->getTH1F();

  //   TH1F *METHFPhiRatio = METHFPhiComp_->getTH1F();

  //   METHFPhiRatio->Divide(methfphiNum, methfphiDen);
  // }

  // MHT
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MHTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MHTRank");


  if (dataHist_ && emulHist_){
    TH1F *mhtNum = dataHist_->getTH1F();
    TH1F *mhtDen = emulHist_->getTH1F();

    TH1F *MHTRatio = MHTComp_->getTH1F();

    MHTRatio->Divide(mhtNum, mhtDen);
  }

  // // MHT Phi
  // dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MHTPhi");
  // emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MHTPhi");

  // if (dataHist_ && emulHist_){
  //   TH1F * mhtphiNum = dataHist_->getTH1F();
  //   TH1F * mhtphiDen = emulHist_->getTH1F();

  //   TH1F * MHTPhiRatio = METHFPhiComp_->getTH1F();

  //   MHTPhiRatio->Divide(mhtphiNum, mhtphiDen);
  // }

  // MHTHF
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MHTHFRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MHTHFRank");

  if (dataHist_ && emulHist_){
    TH1F *mhthfNum = dataHist_->getTH1F();
    TH1F *mhthfDen = emulHist_->getTH1F();

    TH1F *MHTHFRatio = MHTHFComp_->getTH1F();

    MHTHFRatio->Divide(mhthfNum, mhthfDen);
  }

  // // MHTHF Phi
  // dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MHTHFPhi");
  // emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MHTHFPhi");

  // if (dataHist_ && emulHist_){
  //   TH1F *mhthfphiNum = dataHist_->getTH1F();
  //   TH1F *mhthfphiDen = emulHist_->getTH1F();

  //   TH1F *MHTHFPhiRatio = MHTHFPhiComp_->getTH1F();

  //   std::cout << "dividing mhthf phi" << std::endl;
  //   MHTHFPhiRatio->Divide(mhthfphiNum, mhthfphiDen);
  // }

  // ETT
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"ETTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"ETTRank");

  if (dataHist_ && emulHist_){
    TH1F *ettNum = dataHist_->getTH1F();
    TH1F *ettDen = emulHist_->getTH1F();

    TH1F *ETTRatio = ETTComp_->getTH1F();

    ETTRatio->Divide(ettNum, ettDen);
  }

  // ETTEM
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"ETTEMRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"ETTEMRank");

  if (dataHist_ && emulHist_){
    TH1F *ettemNum = dataHist_->getTH1F();
    TH1F *ettemDen = emulHist_->getTH1F();

    TH1F *ETTEMRatio = ETTEMComp_->getTH1F();

    ETTEMRatio->Divide(ettemNum, ettemDen);
  }

  // HTT
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"HTTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"HTTRank");

  if (dataHist_ && emulHist_){
    TH1F *httNum = dataHist_->getTH1F();
    TH1F *httDen = emulHist_->getTH1F();

    TH1F *HTTRatio = HTTComp_->getTH1F();

    HTTRatio->Divide(httNum, httDen);
  }

  // MBHFP0
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MinBiasHFP0");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MinBiasHFP0");

  if (dataHist_ && emulHist_){
    TH1F *mbhfp0Num = dataHist_->getTH1F();
    TH1F *mbhfp0Den = emulHist_->getTH1F();

    TH1F *MBHFP0Ratio = MinBiasHFP0Comp_->getTH1F();

    MBHFP0Ratio->Divide(mbhfp0Num, mbhfp0Den);
  }

  // MBHFM0
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MinBiasHFM0");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MinBiasHFM0");

  if (dataHist_ && emulHist_){
    TH1F *mbhfm0Num = dataHist_->getTH1F();
    TH1F *mbhfm0Den = emulHist_->getTH1F();

    TH1F *MBHFM0Ratio = MinBiasHFM0Comp_->getTH1F();

    MBHFM0Ratio->Divide(mbhfm0Num, mbhfm0Den);
  }

  // MBHFP1
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MinBiasHFP1");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MinBiasHFP1");

  if (dataHist_ && emulHist_){
    TH1F *mbhfp1Num = dataHist_->getTH1F();
    TH1F *mbhfp1Den = emulHist_->getTH1F();

    TH1F *MBHFP1Ratio = MinBiasHFP1Comp_->getTH1F();

    MBHFP1Ratio->Divide(mbhfp1Num, mbhfp1Den);
  }

  // MBHFM1
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MinBiasHFM1");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MinBiasHFM1");

  if (dataHist_ && emulHist_){
    TH1F *mbhfm1Num = dataHist_->getTH1F();
    TH1F *mbhfm1Den = emulHist_->getTH1F();

    TH1F *MBHFM1Ratio = MinBiasHFM1Comp_->getTH1F();

    MBHFM1Ratio->Divide(mbhfm1Num, mbhfm1Den);
  }

  // TowerCount
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"TowCount");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"TowCount");

  if (dataHist_ && emulHist_){
    TH1F *towCountNum = dataHist_->getTH1F();
    TH1F *towCountDen = emulHist_->getTH1F();

    TH1F *TowCountRatio = TowerCountComp_->getTH1F();

    TowCountRatio->Divide(towCountNum, towCountDen);
  }
}
