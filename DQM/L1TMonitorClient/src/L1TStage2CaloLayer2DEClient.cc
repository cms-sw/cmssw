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
}
  
void L1TStage2CaloLayer2DEClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
}


  
