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
  CenJetEtaComp_=ibooker.book1D("CenJetsPhiDERatio","Data/Emul of central jet Phi", 144, -0.5, 143.5);
  ForJetRankComp_=ibooker.book1D("ForJetsRankDERatio","Data/Emul of forward jet E_{T}", 2048, -0.5, 2047.5);
  ForJetEtaComp_=ibooker.book1D("ForJetsEtaDERatio","Data/Emul of forward jet Eta", 229, -114.5, 114.5);
  ForJetEtaComp_=ibooker.book1D("ForJetsPhiDERatio","Data/Emul of forward jet Phi", 144, -0.5, 143.5);
  IsoEGRankComp_=ibooker.book1D("IsoEGRankDERatio","Data/Emul of isolated eg E_{T}", 512, -0.5, 511.5);
  IsoEGEtaComp_=ibooker.book1D("IsoEGEtaDERatio","Data/Emul of isolated eg Eta", 229, -114.5, 114.5);
  IsoEGEtaComp_=ibooker.book1D("IsoEGPhiDERatio","Data/Emul of isolated eg Phi", 144, -0.5, 143.5);
  NonIsoEGRankComp_=ibooker.book1D("NonIsoEGRankDERatio","Data/Emul of non-isolated eg E_{T}", 512, -0.5, 511.5);
  NonIsoEGEtaComp_=ibooker.book1D("NonIsoEGEtaDERatio","Data/Emul of non-isolated eg Eta", 229, -114.5, 114.5);
  NonIsoEGEtaComp_=ibooker.book1D("NonIsoEGPhiDERatio","Data/Emul of non-isolated eg Phi", 144, -0.5, 143.5);
  TauRankComp_=ibooker.book1D("TauRankDERatio","Data/Emul of relax tau E_{T}", 512, -0.5, 511.5);
  TauEtaComp_=ibooker.book1D("TauEtaDERatio","Data/Emul of relax tau Eta", 229, -114.5, 114.5);
  TauEtaComp_=ibooker.book1D("TauPhiDERatio","Data/Emul of relax tau eg Phi", 144, -0.5, 143.5);
  IsoTauRankComp_=ibooker.book1D("IsoTauRankDERatio","Data/Emul of iso tau E_{T}", 512, -0.5, 511.5);
  IsoTauEtaComp_=ibooker.book1D("IsoTauEtaDERatio","Data/Emul of iso tau Eta", 229, -114.5, 114.5);
  IsoTauEtaComp_=ibooker.book1D("IsoTauPhiDERatio","Data/Emul of iso tau eg Phi", 144, -0.5, 143.5);
  METComp_=ibooker.book1D("METRatio","Data/Emul of MET", 4096, -0.5, 4095.5);
  MHTComp_=ibooker.book1D("MHTRatio","Data/Emul of MHT", 4096, -0.5, 4095.5);
  ETTComp_=ibooker.book1D("ETTRatio","Data/Emul of ET Total", 4096, -0.5, 4095.5);
  HTTComp_=ibooker.book1D("HTTRatio","Data/Emul of HT Total", 4096, -0.5, 4095.5);
}

void L1TStage2CaloLayer2DEClient::processHistograms(DQMStore::IGetter &igetter){
  
  MonitorElement* dataHist_;
  MonitorElement* emulHist_;

  // central jets
  dataHist_ = igetter.get(input_dir_data_+"/"+"CenJetsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"CenJetsRank");

  TH1F *cjrNum = dataHist_->getTH1F();
  TH1F *cjrDen = emulHist_->getTH1F();

  TH1F *CenJetRankRatio = CenJetRankComp_->getTH1F();

  CenJetRankRatio->Divide(cjrNum, cjrDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"CenJetsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"CenJetsEta");

  TH1F *cjeNum = dataHist_->getTH1F();
  TH1F *cjeDen = emulHist_->getTH1F();

  TH1F *CenJetEtaRatio = CenJetEtaComp_->getTH1F();

  CenJetEtaRatio->Divide(cjeNum, cjeDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"CenJetsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"CenJetsPhi");

  TH1F *cjpNum = dataHist_->getTH1F();
  TH1F *cjpDen = emulHist_->getTH1F();

  TH1F *CenJetPhiRatio = CenJetPhiComp_->getTH1F();

  CenJetPhiRatio->Divide(cjpNum, cjpDen);

  // forward jets
  dataHist_ = igetter.get(input_dir_data_+"/"+"ForJetsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"ForJetsRank");

  TH1F *fjrNum = dataHist_->getTH1F();
  TH1F *fjrDen = emulHist_->getTH1F();

  TH1F *ForJetRankRatio = ForJetRankComp_->getTH1F();

  ForJetRankRatio->Divide(fjrNum, fjrDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"ForJetsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"ForJetsEta");

  TH1F *fjeNum = dataHist_->getTH1F();
  TH1F *fjeDen = emulHist_->getTH1F();

  TH1F *ForJetEtaRatio = ForJetEtaComp_->getTH1F();

  ForJetEtaRatio->Divide(fjeNum, fjeDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"ForJetsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"ForJetsPhi");

  TH1F *fjpNum = dataHist_->getTH1F();
  TH1F *fjpDen = emulHist_->getTH1F();

  TH1F *ForJetPhiRatio = ForJetPhiComp_->getTH1F();

  ForJetPhiRatio->Divide(fjpNum, fjpDen);  

  // isolated eg
  dataHist_ = igetter.get(input_dir_data_+"/"+"IsoEGsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"IsoEGsRank");

  TH1F *ierNum = dataHist_->getTH1F();
  TH1F *ierDen = emulHist_->getTH1F();

  TH1F *IsoEGRankRatio = IsoEGRankComp_->getTH1F();

  IsoEGRankRatio->Divide(ierNum, ierDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"IsoEGsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"IsoEGsEta");

  TH1F *ieeNum = dataHist_->getTH1F();
  TH1F *ieeDen = emulHist_->getTH1F();

  TH1F *IsoEGEtaRatio = IsoEGEtaComp_->getTH1F();

  IsoEGEtaRatio->Divide(ieeNum, ieeDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"IsoEGsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"IsoEGsPhi");

  TH1F *iepNum = dataHist_->getTH1F();
  TH1F *iepDen = emulHist_->getTH1F();

  TH1F *IsoEGPhiRatio = IsoEGPhiComp_->getTH1F();

  IsoEGPhiRatio->Divide(iepNum, iepDen);

  // non-isolated eg
  dataHist_ = igetter.get(input_dir_data_+"/"+"NonIsoEGsRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"NonIsoEGsRank");

  TH1F *nerNum = dataHist_->getTH1F();
  TH1F *nerDen = emulHist_->getTH1F();

  TH1F *NonIsoEGRankRatio = NonIsoEGRankComp_->getTH1F();

  NonIsoEGRankRatio->Divide(nerNum, nerDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"NonIsoEGsEta");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"NonIsoEGsEta");

  TH1F *neeNum = dataHist_->getTH1F();
  TH1F *neeDen = emulHist_->getTH1F();

  TH1F *NonIsoEGEtaRatio = NonIsoEGEtaComp_->getTH1F();

  NonIsoEGEtaRatio->Divide(neeNum, neeDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"NonIsoEGsPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"NonIsoEGsPhi");

  TH1F *nepNum = dataHist_->getTH1F();
  TH1F *nepDen = emulHist_->getTH1F();

  TH1F *NonIsoEGPhiRatio = NonIsoEGPhiComp_->getTH1F();

  NonIsoEGPhiRatio->Divide(nepNum, nepDen);

  // rlx tau
  dataHist_ = igetter.get(input_dir_data_+"/"+"TausRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"TausRank");

  TH1F *trNum = dataHist_->getTH1F();
  TH1F *trDen = emulHist_->getTH1F();

  TH1F *TauRankRatio = TauRankComp_->getTH1F();

  TauRankRatio->Divide(trNum, trDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"TausEta");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"TausEta");

  TH1F *teNum = dataHist_->getTH1F();
  TH1F *teDen = emulHist_->getTH1F();

  TH1F *TauEtaRatio = TauEtaComp_->getTH1F();

  TauEtaRatio->Divide(teNum, teDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"TausPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"TausPhi");

  TH1F *tpNum = dataHist_->getTH1F();
  TH1F *tpDen = emulHist_->getTH1F();

  TH1F *TauPhiRatio = TauPhiComp_->getTH1F();

  TauPhiRatio->Divide(tpNum, tpDen);  

  // iso tau
  dataHist_ = igetter.get(input_dir_data_+"/"+"IsoTausRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"IsoTausRank");

  TH1F *itrNum = dataHist_->getTH1F();
  TH1F *itrDen = emulHist_->getTH1F();

  TH1F *IsoTauRankRatio = IsoTauRankComp_->getTH1F();

  IsoTauRankRatio->Divide(itrNum, itrDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"IsoTausEta");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"IsoTausEta");

  TH1F *iteNum = dataHist_->getTH1F();
  TH1F *iteDen = emulHist_->getTH1F();

  TH1F *IsoTauEtaRatio = IsoTauEtaComp_->getTH1F();

  IsoTauEtaRatio->Divide(iteNum, iteDen);

  dataHist_ = igetter.get(input_dir_data_+"/"+"IsoTausPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"IsoTausPhi");

  TH1F *itpNum = dataHist_->getTH1F();
  TH1F *itpDen = emulHist_->getTH1F();

  TH1F *IsoTauPhiRatio = IsoTauPhiComp_->getTH1F();

  IsoTauPhiRatio->Divide(itpNum, itpDen);

  // MET
  dataHist_ = igetter.get(input_dir_data_+"/"+"METRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"METRank");

  TH1F *metNum = dataHist_->getTH1F();
  TH1F *metDen = emulHist_->getTH1F();

  TH1F *METRatio = METComp_->getTH1F();

  METRatio->Divide(metNum, metDen);

  // MHT
  dataHist_ = igetter.get(input_dir_data_+"/"+"MHTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"MHTRank");

  TH1F *mhtNum = dataHist_->getTH1F();
  TH1F *mhtDen = emulHist_->getTH1F();

  TH1F *MHTRatio = MHTComp_->getTH1F();

  MHTRatio->Divide(mhtNum, mhtDen);

  // ETT
  dataHist_ = igetter.get(input_dir_data_+"/"+"ETTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"ETTRank");

  TH1F *ettNum = dataHist_->getTH1F();
  TH1F *ettDen = emulHist_->getTH1F();

  TH1F *ETTRatio = ETTComp_->getTH1F();

  ETTRatio->Divide(ettNum, ettDen);

  // HTT
  dataHist_ = igetter.get(input_dir_data_+"/"+"HTTRank");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"HTTRank");

  TH1F *httNum = dataHist_->getTH1F();
  TH1F *httDen = emulHist_->getTH1F();

  TH1F *HTTRatio = HTTComp_->getTH1F();

  HTTRatio->Divide(httNum, httDen);    
}
  
void L1TStage2CaloLayer2DEClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
      book(ibooker);
      processHistograms(igetter);
}


  
