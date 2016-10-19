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

  SummaryPlot_ = ibooker.book1D("CaloLayer2Summary", "CaloLayer2 Data-Emulator agreement summary", 35, 0, 35);
}

void L1TStage2CaloLayer2DEClient::processHistograms(DQMStore::IGetter &igetter){
  
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


  // Summary plot
  TH1F * SummaryHist = SummaryPlot_->getTH1F();
  for (int i = 1; i < SummaryHist->GetXaxis()->GetNbins() + 1; ++i) {
    SummaryHist->GetXaxis()->SetBinLabel(i, summaryLabels[i-1].c_str());
  }

  // CenJetEt, CenJetPhi, CenJetEta, ForJetEt, ForJetPhi, ForJetEta
  // IsoEGEt, IsoEGPhi, IsoEGEta, NIsoEGEt, NIsoEGPhi, NIsoEGEta
  // IsoTauEt, IsoTauPhi, IsoTauEta, NIsoTauEt, NIsoTauPhi, NIsoTauEta, 
  // METRank, METPhi, METHFRank, METHFPhi, 
  // ETTRank, ETTHFRank, HTTRank, HTTHFRank, 


  // temporarily create placeholders for histograms that do not yet exist in DQM

  // central jet et
  addObjToSummary(igetter, SummaryHist, std::string("/Central-Jets/CenJetsRank"), summaryLabels[0].c_str());
  
  // central jet eta
  addObjToSummary(igetter, SummaryHist, std::string("/Central-Jets/CenJetsEta"), summaryLabels[1].c_str());
  
  // central jet phi
  addObjToSummary(igetter, SummaryHist, std::string("/Central-Jets/CenJetsPhi"), summaryLabels[2].c_str());

  // forward jet et
  addObjToSummary(igetter, SummaryHist, std::string("/Forward-Jets/ForJetsRank"), summaryLabels[3].c_str());
  // TH1F *fjrNum = igetter.get(input_dir_data_+"/Forward-Jets/"+"ForJetsRank")->getTH1F();
  // TH1F *fjrDen = igetter.get(input_dir_emul_+"/Forward-Jets/"+"ForJetsRank")->getTH1F();

  // forward jet eta 
  addObjToSummary(igetter, SummaryHist, std::string("/Forward-Jets/ForJetsEta"), summaryLabels[4].c_str());
  // TH1F *fjeNum = igetter.get(input_dir_data_+"/Forward-Jets/"+"ForJetsEta")->getTH1F();
  // TH1F *fjeDen = igetter.get(input_dir_emul_+"/Forward-Jets/"+"ForJetsEta")->getTH1F();

  // foward jet phi
  addObjToSummary(igetter, SummaryHist, std::string("/Forward-Jets/ForJetsPhi"), summaryLabels[5].c_str());
  // TH1F *fjpNum = igetter.get(input_dir_data_+"/Forward-Jets/"+"ForJetsPhi")->getTH1F();
  // TH1F *fjpDen = igetter.get(input_dir_emul_+"/Forward-Jets/"+"ForJetsPhi")->getTH1F();

  // iso eg et
  addObjToSummary(igetter, SummaryHist, std::string("/Isolated-EG/IsoEGsRank"), summaryLabels[6].c_str());
  // TH1F *ierNum = igetter.get(input_dir_data_+"/Isolated-EG/"+"IsoEGsRank")->getTH1F();
  // TH1F *ierDen = igetter.get(input_dir_emul_+"/Isolated-EG/"+"IsoEGsRank")->getTH1F();

  // iso eg eta
  addObjToSummary(igetter, SummaryHist, std::string("/Isolated-EG/IsoEGsEta"), summaryLabels[7].c_str());
  // TH1F *ieeNum = igetter.get(input_dir_data_+"/Isolated-EG/"+"IsoEGsEta")->getTH1F();
  // TH1F *ieeDen = igetter.get(input_dir_emul_+"/Isolated-EG/"+"IsoEGsEta")->getTH1F();

  // iso eg phi
  addObjToSummary(igetter, SummaryHist, std::string("/Isolated-EG/IsoEGsPhi"), summaryLabels[8].c_str());
  // TH1F *iepNum = igetter.get(input_dir_data_+"/Isolated-EG/"+"IsoEGsPhi")->getTH1F();
  // TH1F *iepDen = igetter.get(input_dir_emul_+"/Isolated-EG/"+"IsoEGsPhi")->getTH1F();

  // non iso eg et
  addObjToSummary(igetter, SummaryHist, std::string("/NonIsolated-EG/NonIsoEGsRank"), summaryLabels[9].c_str());
  // TH1F *nerNum = igetter.get(input_dir_data_+"/NonIsolated-EG/"+"NonIsoEGsRank")->getTH1F();
  // TH1F *nerDen = igetter.get(input_dir_emul_+"/NonIsolated-EG/"+"NonIsoEGsRank")->getTH1F();

  // non iso eg eta
  addObjToSummary(igetter, SummaryHist, std::string("/NonIsolated-EG/NonIsoEGsEta"), summaryLabels[10].c_str());
  // TH1F *neeNum = igetter.get(input_dir_data_+"/NonIsolated-EG/"+"NonIsoEGsEta")->getTH1F();
  // TH1F *neeDen = igetter.get(input_dir_emul_+"/NonIsolated-EG/"+"NonIsoEGsEta")->getTH1F();
  
  // non iso eg phi
  addObjToSummary(igetter, SummaryHist, std::string("/NonIsolated-EG/NonIsoEGsPhi"), summaryLabels[11].c_str());
  // TH1F *nepNum = igetter.get(input_dir_data_+"/NonIsolated-EG/"+"NonIsoEGsPhi")->getTH1F();
  // TH1F *nepDen = igetter.get(input_dir_emul_+"/NonIsolated-EG/"+"NonIsoEGsPhi")->getTH1F();

  // iso tau et
  addObjToSummary(igetter, SummaryHist, std::string("/Isolated-Tau/IsoTausRank"), summaryLabels[12].c_str());
  // TH1F *itrNum = igetter.get(input_dir_data_+"/NonIsolated-Tau/"+"IsoTausRank")->getTH1F();
  // TH1F *itrDen = igetter.get(input_dir_emul_+"/NonIsolated-Tau/"+"IsoTausRank")->getTH1F();

  // iso tau eta 
  addObjToSummary(igetter, SummaryHist, std::string("/Isolated-Tau/IsoTausEta"), summaryLabels[13].c_str());
  // TH1F *iteNum = igetter.get(input_dir_data_+"/Isolated-Tau/"+"IsoTausEta")->getTH1F();
  // TH1F *iteDen = igetter.get(input_dir_emul_+"/Isolated-Tau/"+"IsoTausEta")->getTH1F();

  // iso tau phi
  addObjToSummary(igetter, SummaryHist, std::string("/Isolated-Tau/IsoTausPhi"), summaryLabels[14].c_str());
  // TH1F *itpNum = igetter.get(input_dir_data_+"/Isolated-Tau/"+"IsoTausPhi")->getTH1F();
  // TH1F *itpDen = igetter.get(input_dir_emul_+"/Isolated-Tau/"+"IsoTausPhi")->getTH1F();

  // non iso tau et
  addObjToSummary(igetter, SummaryHist, std::string("/NonIsolated-Tau/TausRank"), summaryLabels[15].c_str());
  // TH1F *trNum = igetter.get(input_dir_data_+"/NonIsolated-Tau/"+"TausRank")->getTH1F();
  // TH1F *trDen = igetter.get(input_dir_emul_+"/NonIsolated-Tau/"+"TausRank")->getTH1F();

  // non iso tau phi 
  addObjToSummary(igetter, SummaryHist, std::string("/NonIsolated-Tau/TausPhi"), summaryLabels[16].c_str());
  // TH1F *tpNum = igetter.get(input_dir_data_+"/NonIsolated-Tau/"+"TausPhi")->getTH1F();
  // TH1F *tpDen = igetter.get(input_dir_emul_+"/NonIsolated-Tau/"+"TausPhi")->getTH1F();

  // non iso tau eta 
  addObjToSummary(igetter, SummaryHist, std::string("/NonIsolated-Tau/TausEta"), summaryLabels[17].c_str());
  // TH1F *teNum = igetter.get(input_dir_data_+"/NonIsolated-Tau/"+"TausEta")->getTH1F();
  // TH1F *teDen = igetter.get(input_dir_emul_+"/NonIsolated-Tau/"+"TausEta")->getTH1F();

  // met rank
  addObjToSummary(igetter, SummaryHist, std::string("/Energy-Sums/METRank"), summaryLabels[18].c_str());
  // TH1F *metNum = igetter.get(input_dir_data_+"/Energy-Sums/"+"METRank")->getTH1F();
  // TH1F *metDen = igetter.get(input_dir_emul_+"/Energy-Sums/"+"METRank")->getTH1F();

  // met phi (index - 19)
  // addObjToSummary(igetter, SummaryHist, std::string("/Central-Jets/CenJetsPhi"), summaryLabels[0].c_str());
  // TH1F * metpNum = new TH1F();
  // TH1F * metpDen = new TH1F();

  // methf rank (index - 20)
  // TH1F * methfNum = new TH1F();
  // TH1F * methfDen = new TH1F();

  // methf phi (index - 21)
  // TH1F * methfpNum = new TH1F();
  // TH1F * methfpDen = new TH1F();

  // mht rank (index - 22)
  addObjToSummary(igetter, SummaryHist, std::string("/Energy-Sums/MHTRank"), summaryLabels[22].c_str());
  // TH1F *mhtNum = igetter.get(input_dir_data_+"/Energy-Sums/"+"MHTRank")->getTH1F();
  // TH1F *mhtDen = igetter.get(input_dir_emul_+"/Energy-Sums/"+"MHTRank")->getTH1F();

  // mht phi (index - 23)
  // TH1F * mhtpNum = new TH1F();
  // TH1F * mhtpDen = new TH1F();

  // mhthf rank (index - 24)
  // TH1F * mhthfNum = new TH1F();
  // TH1F * mhthfDen = new TH1F();

  // mhthf phi (index - 25)
  // TH1F * mhthfpNum = new TH1F();
  // TH1F * mhthfpDen = new TH1F();

  // ett (index - 26)
  addObjToSummary(igetter, SummaryHist, std::string("/Energy-Sums/ETTRank"), summaryLabels[26].c_str());
  // TH1F *ettNum = igetter.get(input_dir_data_+"/Energy-Sums/"+"ETTRank")->getTH1F();
  // TH1F *ettDen = igetter.get(input_dir_emul_+"/Energy-Sums/"+"ETTRank")->getTH1F();

  // etthf (index - 27)
  // TH1F * etthfNum = new TH1F();
  // TH1F * etthfDen = new TH1F();

  // htt (index - 28)
  addObjToSummary(igetter, SummaryHist, std::string("/Energy-Sums/HTTRank"), summaryLabels[28].c_str());
  // TH1F *httNum = igetter.get(input_dir_data_+"/Energy-Sums/"+"HTTRank")->getTH1F();
  // TH1F *httDen = igetter.get(input_dir_emul_+"/Energy-Sums/"+"HTTRank")->getTH1F();

  // htthf (index - 29)
  // TH1F * htthfNum = new TH1F();
  // TH1F * htthfDen = new TH1F();

  // mbhfp0 (index - 30)
  // TH1F * mbhfp0Num = new TH1F();
  // TH1F * mbhfp0Den = new TH1F();

  // mbhfm0 (index - 31)
  // TH1F * mbhfm0Num = new TH1F();
  // TH1F * mbhfm0Den = new TH1F();

  // mbhfp1 (index - 32)
  // TH1F * mbhfp1Num = new TH1F();
  // TH1F * mbhfp1Den = new TH1F();

  // mbhfm1 (index - 33)
  // TH1F * mbhfm1Num = new TH1F();
  // TH1F * mbhfm1Den = new TH1F();

  // ettem (index - 34)
  // TH1F * ettemNum = new TH1F();
  // TH1F * ettemDen = new TH1F();

 // populate summary histogram with data from comparisons

  // SummaryHist->GetPainter()->

  SummaryHist->GetXaxis()->SetLabelSize(0.02);
  SummaryHist->SetMarkerStyle(21);

}
void L1TStage2CaloLayer2DEClient::addObjToSummary(DQMStore::IGetter & getter, TH1F * hist, std::string objPath, const char * binLabel) {// ,
  //		     std::string inputDirData, std::string inputDirEmu) {
  
  MonitorElement* dataHist_;
  MonitorElement* emulHist_;

  dataHist_ = getter.get(input_dir_data_ + objPath);
  emulHist_ = getter.get(input_dir_emul_ + objPath);

  if (dataHist_ && emulHist_){

    double dataInt = dataHist_->getTH1F()->Integral();
    double emuInt = emulHist_->getTH1F()->Integral();
    
    hist->Fill(binLabel, dataInt/emuInt);
  }

}  
void L1TStage2CaloLayer2DEClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}


  
