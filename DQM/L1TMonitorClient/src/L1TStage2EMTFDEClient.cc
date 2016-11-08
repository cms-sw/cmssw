#include "DQM/L1TMonitorClient/interface/L1TStage2EMTFDEClient.h"

L1TStage2EMTFDEClient::L1TStage2EMTFDEClient(const edm::ParameterSet& ps):
  monitor_dir_(ps.getUntrackedParameter<std::string>("monitorDir","")),
  input_dir_data_(ps.getUntrackedParameter<std::string>("inputDataDir","")),
  input_dir_emul_(ps.getUntrackedParameter<std::string>("inputEmulDir",""))
{}

L1TStage2EMTFDEClient::~L1TStage2EMTFDEClient(){}

void L1TStage2EMTFDEClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,DQMStore::IGetter &igetter,const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) {
  book(ibooker);
  processHistograms(igetter);
}

void L1TStage2EMTFDEClient::book(DQMStore::IBooker &ibooker){
  
  ibooker.setCurrentFolder(monitor_dir_);
  //RegionMuonCand Plots
  emtfMuonhwEtaComp_=ibooker.book1D("emtfMuonhwEtaComp","2*Data/(Emul+Data) of Output Eta", 460, -230,230);
  emtfMuonhwPhiComp_=ibooker.book1D("emtfMuonhwPhiComp", "2*Data/(Emul+Data) of Output Phi", 125, -20, 105);
  emtfMuonhwPtComp_=ibooker.book1D("emtfMuonhwPtComp", "2*Data/(Emul+Data) of Output p_{T}", 512, 0, 512);
  emtfMuonhwQualComp_=ibooker.book1D("emtfMuonhwQualComp", "2*Data/(Emul+Data) of Output Quality", 16, 0, 16);
  emtfMuonhwBXComp_=ibooker.book1D("emtfMuonhwBXComp", "2*Data/(Emul+Data) of Output BX", 7, -3, 4); 

  emtfMuonhwEtaCompCoarse_ = ibooker.book1D("emtfMuonhwEtaCompCoarse", "2*Data/(Emul+Data) of Output Eta (coarse)", 46, -230, 230);
  emtfMuonhwPhiCompCoarse_=ibooker.book1D("emtfMuonhwPhiCompCoarse", "2*Data/(Emul+Data) of Output Phi (coarse)", 25, -20, 105);
  emtfMuonhwPtCompCoarse_=ibooker.book1D("emtfMuonhwPtCompCoarse", "2*Data/(Emul+Data) of Output p_{T} (coarse)", 32, 0, 512);



  for (int bin = 1; bin <= 16; ++bin) {
    emtfMuonhwQualComp_->setBinLabel(bin, std::to_string(bin - 1), 1);
  }
  for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
    emtfMuonhwBXComp_->setBinLabel(bin, std::to_string(bin_label), 1);
  }
  //Track Ratio Plots
  emtfTrackEtaComp_=ibooker.book1D("emtfTrackEtaComp", "2*Data/(Emul+Data) of Track Eta",100,-2.5,2.5);
  emtfTrackPhiComp_=ibooker.book1D("emtfTrackPhiComp", "2*Data/(Emul+Data) of Track Phi", 128, -3.2, 3.2);
  emtfTrackPtComp_=ibooker.book1D("emtfTrackPtComp", "2*Data/(Emul+Data) of Track p_{T}", 256, 1, 257);
  emtfTrackQualComp_=ibooker.book1D("emtfTrackQualComp", "2*Data/(Emul+Data) of Track Quality", 16, 0, 16);
  emtfTrackModeComp_=ibooker.book1D("emtfTrackModeComp", "2*Data/(Emul+Data) of Track Mode", 16, 0, 16);
  emtfTrackBXComp_=ibooker.book1D("emtfTrackBXComp", "2*Data/(Emul+Data) of Track BX", 8, -3, 5);
  
  emtfTrackEtaCompCoarse_=ibooker.book1D("emtfTrackEtaCompCoarse", "2*Data/(Emul+Data) of Track Eta (coarse)",20,-2.5,2.5);
  emtfTrackPhiCompCoarse_=ibooker.book1D("emtfTrackPhiCompCoarse", "2*Data/(Emul+Data) of Track Phi (coarse)", 32, -3.2, 3.2);
  emtfTrackPtCompCoarse_=ibooker.book1D("emtfTrackPtCompCoarse", "2*Data/(Emul+Data) of Track p_{T} (coarse)", 32, 1, 257);
  

  for (int bin = 1, bin_label = -3; bin <= 8; ++bin, ++bin_label) {
    emtfTrackBXComp_->setBinLabel(bin, std::to_string(bin_label), 1);
  }
  emtfTrackSectorIndexComp_=ibooker.book1D("emtfTrackSectorIndexComp","Data/Emul of Track Sector Index", 13, -6.5, 6.5);


  for (int bin = 1; bin <= 16; ++bin) {
    emtfTrackModeComp_->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackQualComp_->setBinLabel(bin, std::to_string(bin - 1), 1);
  }
} 
void L1TStage2EMTFDEClient::fixZero(TH1F* histData, TH1F* histEmul){
  for (Int_t i=0;i<histData->GetSize();i++){
    if(histData->GetBinContent(i)==0) histData->SetBinContent(i,0.01);
  }
  for (Int_t i=0;i<histEmul->GetSize();i++){
    if(histEmul->GetBinContent(i)==0) histEmul->SetBinContent(i,0.01);
  }
} 
void L1TStage2EMTFDEClient::processHistograms(DQMStore::IGetter &igetter){
  
  MonitorElement* dataHist_;
  MonitorElement* emulHist_;

  //Ratio Plots are 2*data/(data + emul) so plots go from 0 to 2, where 0=no data, 1=perfect match, 2=no emul
  //RegionaMuonCand Ratio Plots

  //hwEta
  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonhwEta");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonhwEta");

 if (dataHist_ && emulHist_){
    TH1F *hwEtaD = dataHist_->getTH1F();
    TH1F *hwEtaE = emulHist_->getTH1F();

    TH1F *hwEtaRatio = emtfMuonhwEtaComp_->getTH1F();
    TH1F *hwEtaDif = (TH1F*) hwEtaRatio->Clone();
    L1TStage2EMTFDEClient::fixZero(hwEtaD, hwEtaE); 

    hwEtaDif->Add(hwEtaD, hwEtaE);

    hwEtaRatio->Divide(hwEtaD, hwEtaDif, 2, 1); 
}
  //hwEtaCoarse
  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonhwEtaCoarse");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonhwEtaCoarse");

 if (dataHist_ && emulHist_){
    TH1F *hwEtaCD = dataHist_->getTH1F();
    TH1F *hwEtaCE = emulHist_->getTH1F();

    TH1F *hwEtaRatioCoarse = emtfMuonhwEtaCompCoarse_->getTH1F();
    TH1F *hwEtaDifC = (TH1F*) hwEtaRatioCoarse->Clone();
    L1TStage2EMTFDEClient::fixZero(hwEtaCD, hwEtaCE); 

    hwEtaDifC->Add(hwEtaCD, hwEtaCE);

    hwEtaRatioCoarse->Divide(hwEtaCD, hwEtaDifC, 2, 1); 
}

  //hwPhi
  
  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonhwPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonhwPhi");

 if (dataHist_ && emulHist_){
    TH1F *hwPhiD = dataHist_->getTH1F();
    TH1F *hwPhiE = emulHist_->getTH1F();

    TH1F *hwPhiRatio = emtfMuonhwPhiComp_->getTH1F();
    TH1F *hwPhiDif = (TH1F*) hwPhiRatio->Clone();
    L1TStage2EMTFDEClient::fixZero(hwPhiD, hwPhiE); 
    hwPhiDif->Add(hwPhiD, hwPhiE); 
    hwPhiRatio->Divide(hwPhiD, hwPhiDif, 2, 1);
 }
  //hwPhiCoarse
  
  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonhwPhiCoarse");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonhwPhiCoarse");

 if (dataHist_ && emulHist_){
    TH1F *hwPhiCD = dataHist_->getTH1F();
    TH1F *hwPhiCE = emulHist_->getTH1F();

    TH1F *hwPhiRatioC = emtfMuonhwPhiCompCoarse_->getTH1F();
    TH1F *hwPhiDifC = (TH1F*) hwPhiRatioC->Clone();
    L1TStage2EMTFDEClient::fixZero(hwPhiCD, hwPhiCE); 
    hwPhiDifC->Add(hwPhiCD, hwPhiCE); 
    hwPhiRatioC->Divide(hwPhiCD, hwPhiDifC, 2, 1);
 }

  //hwPt

  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonhwPt");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonhwPt");

 if (dataHist_ && emulHist_){
    TH1F *hwPtD = dataHist_->getTH1F();
    TH1F *hwPtE = emulHist_->getTH1F();

    TH1F *hwPtRatio = emtfMuonhwPtComp_->getTH1F();
    TH1F *hwPtDif = (TH1F*) hwPtRatio->Clone();
  
    L1TStage2EMTFDEClient::fixZero(hwPtD, hwPtE); 
 
    hwPtDif->Add(hwPtD, hwPtE);

    hwPtRatio->Divide(hwPtD, hwPtDif, 2, 1);
 }
 //hwPtCoarse

  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonhwPtCoarse");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonhwPtCoarse");

 if (dataHist_ && emulHist_){
    TH1F *hwPtCD = dataHist_->getTH1F();
    TH1F *hwPtCE = emulHist_->getTH1F();

    TH1F *hwPtRatioC = emtfMuonhwPtCompCoarse_->getTH1F();
    TH1F *hwPtDifC = (TH1F*) hwPtRatioC->Clone();
  
    L1TStage2EMTFDEClient::fixZero(hwPtCD, hwPtCE); 
 
    hwPtDifC->Add(hwPtCD, hwPtCE);

    hwPtRatioC->Divide(hwPtCD, hwPtDifC, 2, 1);
 }

  //hwQual

  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonhwQual");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonhwQual");

 if (dataHist_ && emulHist_){
    TH1F *hwQualD = dataHist_->getTH1F();
    TH1F *hwQualE = emulHist_->getTH1F();

    TH1F *hwQualRatio = emtfMuonhwQualComp_->getTH1F();
    TH1F *hwQualDif = (TH1F*) hwQualRatio->Clone();

 
    //L1TStage2EMTFDEClient::fixZero(hwQualD, hwQualE); 
    
    hwQualDif->Add(hwQualD, hwQualE);
    hwQualRatio->Divide(hwQualD, hwQualDif, 2, 1);
 }
  //hwBX
 
  dataHist_ = igetter.get(input_dir_data_+"/MuonCand/"+"emtfMuonBX");
  emulHist_ = igetter.get(input_dir_emul_+"/MuonCand/"+"emtfMuonBX");

 if (dataHist_ && emulHist_){
    TH1F *hwBXD = dataHist_->getTH1F();
    TH1F *hwBXE = emulHist_->getTH1F();

    TH1F *hwBXRatio = emtfMuonhwBXComp_->getTH1F();
    TH1F *hwBXDif = (TH1F*) hwBXRatio->Clone();


    hwBXDif->Add(hwBXD,hwBXE);
    hwBXRatio->Divide(hwBXD, hwBXDif,2,1);
  }


  //Track Ratio Plots

  //Eta
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackEta");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackEta");

 if (dataHist_ && emulHist_){
    TH1F *EtaD = dataHist_->getTH1F();
    TH1F *EtaE = emulHist_->getTH1F();

    TH1F *EtaRatio = emtfTrackEtaComp_->getTH1F();
    TH1F *EtaDif = (TH1F*) EtaRatio->Clone();

    L1TStage2EMTFDEClient::fixZero(EtaD, EtaE); 
    
    EtaDif->Add(EtaD, EtaE);
   
    EtaRatio->Divide(EtaD, EtaDif, 2, 1); 
  }

   //Eta
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackEtaCoarse");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackEtaCoarse");

 if (dataHist_ && emulHist_){
    TH1F *EtaCD = dataHist_->getTH1F();
    TH1F *EtaCE = emulHist_->getTH1F();

    TH1F *EtaRatioC = emtfTrackEtaCompCoarse_->getTH1F();
    TH1F *EtaDifC = (TH1F*) EtaRatioC->Clone();

    L1TStage2EMTFDEClient::fixZero(EtaCD, EtaCE); 
    
    EtaDifC->Add(EtaCD, EtaCE);
   
    EtaRatioC->Divide(EtaCD, EtaDifC, 2, 1); 
  }

  //Phi
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackPhi");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackPhi");

 if (dataHist_ && emulHist_){
    TH1F *PhiD = dataHist_->getTH1F();
    TH1F *PhiE = emulHist_->getTH1F();

    TH1F *PhiRatio = emtfTrackPhiComp_->getTH1F();
    TH1F *PhiDif = (TH1F*) PhiRatio->Clone();

    L1TStage2EMTFDEClient::fixZero(PhiD, PhiE); 
    
    PhiDif->Add(PhiD, PhiE);
    PhiRatio->Divide(PhiD, PhiDif, 2, 1);
 }
 //Phi
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackPhiCoarse");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackPhiCoarse");

 if (dataHist_ && emulHist_){
    TH1F *PhiCD = dataHist_->getTH1F();
    TH1F *PhiCE = emulHist_->getTH1F();

    TH1F *PhiRatioC = emtfTrackPhiCompCoarse_->getTH1F();
    TH1F *PhiDifC = (TH1F*) PhiRatioC->Clone();

    L1TStage2EMTFDEClient::fixZero(PhiCD, PhiCE); 
    
    PhiDifC->Add(PhiCD, PhiCE);
    PhiRatioC->Divide(PhiCD, PhiDifC, 2, 1);
 }


   //Pt
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackPt");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackPt");

 if (dataHist_ && emulHist_){
    TH1F *PtD = dataHist_->getTH1F();
    TH1F *PtE = emulHist_->getTH1F();

    TH1F *PtRatio = emtfTrackPtComp_->getTH1F(); 
    TH1F *PtDif = (TH1F*) PtRatio->Clone();

    L1TStage2EMTFDEClient::fixZero(PtD, PtE); 
    
    PtDif->Add(PtD, PtE);
    PtRatio->Divide(PtD, PtDif, 2, 1);
 }

  //PtCoarse
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackPtCoarse");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackPtCoarse");

 if (dataHist_ && emulHist_){
    TH1F *PtCD = dataHist_->getTH1F();
    TH1F *PtCE = emulHist_->getTH1F();

    TH1F *PtRatioC = emtfTrackPtCompCoarse_->getTH1F(); 
    TH1F *PtDifC = (TH1F*) PtRatioC->Clone();

    L1TStage2EMTFDEClient::fixZero(PtCD, PtCE); 
    
    PtDifC->Add(PtCD, PtCE);
    PtRatioC->Divide(PtCD, PtDifC, 2, 1);
 }
   //Mode
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackMode");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackMode");

 if (dataHist_ && emulHist_){
    TH1F *ModeD = dataHist_->getTH1F();
    TH1F *ModeE = emulHist_->getTH1F();

    TH1F *ModeRatio = emtfTrackModeComp_->getTH1F();
    TH1F *ModeDif = (TH1F*) ModeRatio->Clone();

    //L1TStage2EMTFDEClient::fixZero(ModeD, ModeE); 
    
    ModeDif->Add(ModeD, ModeE);
    ModeRatio->Divide(ModeD, ModeDif, 2, 1);
 }


   //Quality
  dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackQuality");
  emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackQuality");

 if (dataHist_ && emulHist_){
    TH1F *QualD = dataHist_->getTH1F();
    TH1F *QualE = emulHist_->getTH1F();

    TH1F *QualRatio = emtfTrackQualComp_->getTH1F();
    TH1F *QualDif = (TH1F*) QualRatio->Clone();

    QualDif->Add(QualD, QualE);

    QualRatio->Divide(QualD, QualDif, 2, 1);
 }


   //BX
   dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackBX1D");
   emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackBX1D");

 if (dataHist_ && emulHist_){
    TH1F *BXD = dataHist_->getTH1F();
    TH1F *BXE = emulHist_->getTH1F();

    TH1F *BXRatio = emtfTrackBXComp_->getTH1F();
    TH1F *BXDif = (TH1F*) BXRatio->Clone();

    //L1TStage2EMTFDEClient::fixZero(BXD, BXE); 
    BXDif->Add(BXD, BXE, 1, 1);
    BXRatio->Divide(BXD,BXDif, 2, 1);
 }
  //Sector Index (Sector*Endcap)
   dataHist_ = igetter.get(input_dir_data_+"/"+"emtfTrackSectorIndex");
   emulHist_ = igetter.get(input_dir_emul_+"/"+"emtfTrackSectorIndex");

 if (dataHist_ && emulHist_){
    TH1F *SID = dataHist_->getTH1F();
    TH1F *SIE = emulHist_->getTH1F();

    TH1F *SIRatio = emtfTrackSectorIndexComp_->getTH1F();
    TH1F *SIDif = (TH1F*) SIRatio->Clone();
    
    L1TStage2EMTFDEClient::fixZero(SID, SIE); 
    SIDif->Add(SID, SIE);  
    SIRatio->Divide(SID, SIDif, 2, 1);
 }

}
 
void L1TStage2EMTFDEClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker);
  processHistograms(igetter);
} 
