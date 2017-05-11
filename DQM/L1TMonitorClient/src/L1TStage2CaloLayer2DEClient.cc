#include "DQM/L1TMonitorClient/interface/L1TStage2CaloLayer2DEClient.h"

L1TStage2CaloLayer2DEClient::L1TStage2CaloLayer2DEClient(const edm::ParameterSet& ps):
  monitor_dir_(ps.getUntrackedParameter<std::string>("monitorDir","")),
  input_dir_data_(ps.getUntrackedParameter<std::string>("inputDataDir","")),
  input_dir_emul_(ps.getUntrackedParameter<std::string>("inputEmulDir",""))
{}

L1TStage2CaloLayer2DEClient::~L1TStage2CaloLayer2DEClient(){}

void L1TStage2CaloLayer2DEClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,DQMStore::IGetter &igetter,const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) {
  book(ibooker);
  processHistograms(igetter);
}

void L1TStage2CaloLayer2DEClient::book(DQMStore::IBooker &ibooker){

  ibooker.setCurrentFolder("L1TEMU/L1TdeStage2CaloLayer2");
  hlSummary = ibooker.book1D(
    "High level summary", "Event by event comparison summary", 5, 1, 6);
  hlSummary->getTH1F()->GetXaxis()->SetBinLabel(1, "good events");
  hlSummary->getTH1F()->GetXaxis()->SetBinLabel(2, "good jets");
  hlSummary->getTH1F()->GetXaxis()->SetBinLabel(3, "good e/gs");
  hlSummary->getTH1F()->GetXaxis()->SetBinLabel(4, "good taus");
  hlSummary->getTH1F()->GetXaxis()->SetBinLabel(5, "good sums");

  jetSummary = ibooker.book1D(
    "Jet Agreement Summary", "Jet Agreement Summary", 3, 1, 4);
  jetSummary->getTH1F()->GetXaxis()->SetBinLabel(1, "good jets");
  jetSummary->getTH1F()->GetXaxis()->SetBinLabel(2, "jets pos off only");
  jetSummary->getTH1F()->GetXaxis()->SetBinLabel(3, "jets Et off only ");

  egSummary = ibooker.book1D(
    "EG Agreement Summary", "EG Agreement Summary", 6, 1, 7);
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(1, "good non-iso e/gs");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(2, "non-iso e/gs pos off");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(3, "non-iso e/gs Et off");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(4, "good iso e/gs");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(5, "iso e/gs pos off");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(6, "iso e/gs Et off");

  tauSummary = ibooker.book1D(
    "Tau Agreement Summary", "Tau Agremeent Summary", 6, 1, 7);
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(1, "good non-iso taus");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(2, "non-iso taus pos off");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(3, "non-iso taus Et off");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(4, "good iso taus");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(5, "iso taus pos off");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(6, "iso taus Et off");

  sumSummary = ibooker.book1D(
    "Energy Sum Agreement Summary", "Sum Agreement Summary", 7, 1, 8);
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(1, "good sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(2, "good ETT sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(3, "good HTT sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(4, "good MET sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(5, "good MHT sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(6, "good MBHF sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(7, "good TowCount sums");

  // problemSummary;

  ibooker.setCurrentFolder(monitor_dir_);

  CenJetRankComp_=ibooker.book1D("CenJetsRankDERatio","Data/Emul of central jet E_{T}", 2048, -0.5, 2047.5);
  CenJetEtaComp_=ibooker.book1D("CenJetsEtaDERatio","Data/Emul of central jet Eta", 229, -114.5, 114.5);
  CenJetPhiComp_=ibooker.book1D("CenJetsPhiDERatio","Data/Emul of central jet Phi", 144, -0.5, 143.5);
  ForJetRankComp_=ibooker.book1D("ForJetsRankDERatio","Data/Emul of forward jet E_{T}", 2048, -0.5, 2047.5);
  ForJetEtaComp_=ibooker.book1D("ForJetsEtaDERatio","Data/Emul of forward jet Eta", 229, -114.5, 114.5);
  ForJetPhiComp_=ibooker.book1D("ForJetsPhiDERatio","Data/Emul of forward jet Phi", 144, -0.5, 143.5);
  IsoEGRankComp_=ibooker.book1D("IsoEGRankDERatio","Data/Emul of isolated eg E_{T}", 512, -0.5, 511.5);
  IsoEGEtaComp_=ibooker.book1D("IsoEGEtaDERatio","Data/Emul of isolated eg Eta", 229, -114.5, 114.5);
  IsoEGPhiComp_=ibooker.book1D("IsoEGPhiDERatio","Data/Emul of isolated eg Phi", 144, -0.5, 143.5);
  NonIsoEGRankComp_=ibooker.book1D("NonIsoEGRankDERatio","Data/Emul of non-isolated eg E_{T}", 512, -0.5, 511.5);
  NonIsoEGEtaComp_=ibooker.book1D("NonIsoEGEtaDERatio","Data/Emul of non-isolated eg Eta", 229, -114.5, 114.5);
  NonIsoEGPhiComp_=ibooker.book1D("NonIsoEGPhiDERatio","Data/Emul of non-isolated eg Phi", 144, -0.5, 143.5);
  TauRankComp_=ibooker.book1D("TauRankDERatio","Data/Emul of relax tau E_{T}", 512, -0.5, 511.5);
  TauEtaComp_=ibooker.book1D("TauEtaDERatio","Data/Emul of relax tau Eta", 229, -114.5, 114.5);
  TauPhiComp_=ibooker.book1D("TauPhiDERatio","Data/Emul of relax tau eg Phi", 144, -0.5, 143.5);
  IsoTauRankComp_=ibooker.book1D("IsoTauRankDERatio","Data/Emul of iso tau E_{T}", 512, -0.5, 511.5);
  IsoTauEtaComp_=ibooker.book1D("IsoTauEtaDERatio","Data/Emul of iso tau Eta", 229, -114.5, 114.5);
  IsoTauPhiComp_=ibooker.book1D("IsoTauPhiDERatio","Data/Emul of iso tau eg Phi", 144, -0.5, 143.5);
  METComp_=ibooker.book1D("METRatio","Data/Emul of MET", 4096, -0.5, 4095.5);
  MHTComp_=ibooker.book1D("MHTRatio","Data/Emul of MHT", 4096, -0.5, 4095.5);
  ETTComp_=ibooker.book1D("ETTRatio","Data/Emul of ET Total", 4096, -0.5, 4095.5);
  HTTComp_=ibooker.book1D("HTTRatio","Data/Emul of HT Total", 4096, -0.5, 4095.5);


}

void L1TStage2CaloLayer2DEClient::processHistograms(DQMStore::IGetter &igetter){

  // get reference to relevant summary MonitorElement instances
  // - high level summary
  // - eg agreement summary
  // - energy sum agreement summary
  // - jet agreement summary
  // - tau agreement summary

  TH1F * hist;
  TH1F * newHist;

  MonitorElement * hlSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/CaloL2 Object Agreement Summary");
  MonitorElement * jetSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/Jet Agreement Summary");
  MonitorElement * egSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/EG Agreement Summary");
  MonitorElement * tauSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/Tau Agreement Summary");
  MonitorElement * sumSummary_ = igetter.get(
    "L1TEMU/L1TdeStage2CaloLayer2/Emergy Sum Agreement Summary");

  // check for existance of object
  if (hlSummary) {
    std::cout << "meh: hl histogram found." << std::endl;

    // reference the histogram in MonitorElement
    hist = hlSummary_->getTH1F();
    newHist = hlSummary->getTH1F();

    double totalEvents = 0, goodEvents = 0, totalJets = 0, goodJets = 0,
      totalEg = 0, goodEg = 0, totalTau = 0, goodTau = 0, totalSums = 0,
      goodSums = 0;

    double evtRatio = 0, jetRatio = 0, egRatio = 0, tauRatio = 0, sumRatio = 0;

    totalEvents = hist->GetBinContent(1);
    goodEvents  = hist->GetBinContent(2);
    totalJets   = hist->GetBinContent(3);
    goodJets    = hist->GetBinContent(4);
    totalEg     = hist->GetBinContent(5);
    goodEg      = hist->GetBinContent(6);
    totalTau    = hist->GetBinContent(7);
    goodTau     = hist->GetBinContent(8);
    totalSums   = hist->GetBinContent(9);
    goodSums    = hist->GetBinContent(10);

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

    newHist->SetBinContent(newHist->FindBin(1), evtRatio);
    newHist->SetBinContent(newHist->FindBin(2), jetRatio);
    newHist->SetBinContent(newHist->FindBin(3), egRatio);
    newHist->SetBinContent(newHist->FindBin(4), tauRatio);
    newHist->SetBinContent(newHist->FindBin(5), sumRatio);
  }

  if (jetSummary_) {

    double totalJets = 0, goodJets = 0, jetPosOff = 0, jetEtOff = 0;

    double goodRatio = 0, posOffRatio = 0, etOffRatio = 0;

    hist = jetSummary_->getTH1F();
    newHist = jetSummary->getTH1F();

    totalJets = hist->GetBinContent(1);
    goodJets  = hist->GetBinContent(2);
    jetPosOff = hist->GetBinContent(3);
    jetEtOff  = hist->GetBinContent(4);

    if (totalJets != 0) {
      goodRatio = goodJets / totalJets;
      posOffRatio = jetPosOff / totalJets;
      etOffRatio = jetEtOff / totalJets;
    }

    newHist->SetBinContent(newHist->FindBin(1), goodRatio);
    newHist->SetBinContent(newHist->FindBin(2), posOffRatio);
    newHist->SetBinContent(newHist->FindBin(3), etOffRatio);
  }

  if (egSummary_) {

    double totalEgs = 0, goodEgs = 0, egPosOff = 0, egEtOff = 0,
      totalIsoEgs = 0, goodIsoEgs = 0, isoEgPosOff = 0, isoEgEtOff = 0;

    double goodEgRatio = 0, egPosOffRatio = 0, egEtOffRatio = 0,
      goodIsoEgRatio = 0, isoEgPosOffRatio = 0, isoEgEtOffRatio = 0;

    hist = egSummary_->getTH1F();
    newHist = tauSummary->getTH1F();

    totalEgs = hist->GetBinContent(1);
    goodEgs  = hist->GetBinContent(2);
    egPosOff = hist->GetBinContent(3);
    egEtOff  = hist->GetBinContent(4);

    totalIsoEgs = hist->GetBinContent(5);
    goodIsoEgs  = hist->GetBinContent(6);
    isoEgPosOff = hist->GetBinContent(7);
    isoEgEtOff  = hist->GetBinContent(8);

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

    newHist->SetBinContent(newHist->FindBin(1), goodEgRatio);
    newHist->SetBinContent(newHist->FindBin(2), egPosOffRatio);
    newHist->SetBinContent(newHist->FindBin(3), egEtOffRatio);

    newHist->SetBinContent(newHist->FindBin(4), goodIsoEgRatio);
    newHist->SetBinContent(newHist->FindBin(5), isoEgPosOffRatio);
    newHist->SetBinContent(newHist->FindBin(6), isoEgEtOffRatio);
  }

  if (tauSummary_) {

    double totalTaus = 0, goodTaus = 0, tauPosOff = 0, tauEtOff = 0,
      totalIsoTaus = 0, goodIsoTaus = 0, isoTauPosOff = 0, isoTauEtOff = 0;

    double goodTauRatio = 0, tauPosOffRatio = 0, tauEtOffRatio = 0,
      goodIsoTauRatio = 0, isoTauPosOffRatio= 0, isoTauEtOffRatio = 0;

    hist = tauSummary_->getTH1F();
    newHist = tauSummary->getTH1F();

    totalTaus = hist->GetBinContent(1);
    goodTaus  = hist->GetBinContent(2);
    tauPosOff = hist->GetBinContent(3);
    tauEtOff  = hist->GetBinContent(4);

    totalIsoTaus = hist->GetBinContent(5);
    goodIsoTaus  = hist->GetBinContent(6);
    isoTauPosOff = hist->GetBinContent(7);
    isoTauEtOff  = hist->GetBinContent(8);

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

    newHist->SetBinContent(newHist->FindBin(1), goodTauRatio);
    newHist->SetBinContent(newHist->FindBin(2), tauPosOffRatio);
    newHist->SetBinContent(newHist->FindBin(3), tauEtOffRatio);

    newHist->SetBinContent(newHist->FindBin(4), goodIsoTauRatio);
    newHist->SetBinContent(newHist->FindBin(5), isoTauPosOffRatio);
    newHist->SetBinContent(newHist->FindBin(6), isoTauEtOffRatio);

  }

  if (sumSummary_) {

    double totalSums = 0, goodSums = 0, totalETT = 0, goodETT = 0, totalHTT = 0,
      goodHTT = 0, totalMET = 0, goodMET = 0, totalMHT = 0, goodMHT = 0,
      totalMBHF = 0, goodMBHF = 0, totalTowCount = 0, goodTowCount = 0;

    double goodSumRatio = 0, goodETTRatio = 0, goodHTTRatio = 0,
      goodMETRatio = 0, goodMHTRatio = 0, goodMBHFRatio = 0,
      goodTowCountRatio = 0;

    hist = sumSummary_->getTH1F();
    newHist = sumSummary->getTH1F();

    totalSums     = hist->GetBinContent(1);
    goodSums      = hist->GetBinContent(2);
    totalETT      = hist->GetBinContent(3);
    goodSums      = hist->GetBinContent(4);
    totalHTT      = hist->GetBinContent(5);
    goodHTT       = hist->GetBinContent(6);
    totalMET      = hist->GetBinContent(7);
    goodMET       = hist->GetBinContent(8);
    totalMHT      = hist->GetBinContent(9);
    goodMHT       = hist->GetBinContent(10);
    totalMBHF     = hist->GetBinContent(11);
    goodMBHF      = hist->GetBinContent(12);
    totalTowCount = hist->GetBinContent(13);
    goodTowCount  = hist->GetBinContent(14);

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

    newHist->SetBinContent(newHist->FindBin(1), goodSumRatio);
    newHist->SetBinContent(newHist->FindBin(2), goodETTRatio);
    newHist->SetBinContent(newHist->FindBin(3), goodHTTRatio);
    newHist->SetBinContent(newHist->FindBin(4), goodMETRatio);
    newHist->SetBinContent(newHist->FindBin(5), goodMHTRatio);
    newHist->SetBinContent(newHist->FindBin(6), goodMBHFRatio);
    newHist->SetBinContent(newHist->FindBin(7), goodTowCountRatio);
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

  // MHT
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"MHTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MHTRank");

  if (dataHist_ && emulHist_){
    TH1F *mhtNum = dataHist_->getTH1F();
    TH1F *mhtDen = emulHist_->getTH1F();

    TH1F *MHTRatio = MHTComp_->getTH1F();
    
    MHTRatio->Divide(mhtNum, mhtDen);
  }

  // ETT
  dataHist_ = igetter.get(input_dir_data_+"/Energy-Sums/"+"ETTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/Energy-Sums/"+"ETTRank");

  if (dataHist_ && emulHist_){
    TH1F *ettNum = dataHist_->getTH1F();
    TH1F *ettDen = emulHist_->getTH1F();
    
    TH1F *ETTRatio = ETTComp_->getTH1F();
    
    ETTRatio->Divide(ettNum, ettDen);
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
}
  
void L1TStage2CaloLayer2DEClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}

