// -*- C++ -*-
//
// Package:    DQMOffline/JetMET
// Class:      DataCertificationJetMET
// 
// Original Author:  "Frank Chlebana"
//         Created:  Sun Oct  5 13:57:25 CDT 2008
//

#include "DQMOffline/JetMET/interface/DataCertificationJetMET.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

// Some switches
//
// constructors and destructor
//
DataCertificationJetMET::DataCertificationJetMET(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  // now do what ever initialization is needed
  inputMETLabelRECO_=iConfig.getParameter<edm::InputTag>("METTypeRECO");
  inputMETLabelMiniAOD_=iConfig.getParameter<edm::InputTag>("METTypeMiniAOD");
  inputJetLabelRECO_=iConfig.getParameter<edm::InputTag>("JetTypeRECO");
  inputJetLabelMiniAOD_=iConfig.getParameter<edm::InputTag>("JetTypeMiniAOD");

  nbinsPV_ = iConfig.getParameter<int>("pVBin");
  nPVMin_  = iConfig.getParameter<double>("pVMin");
  nPVMax_  = iConfig.getParameter<double>("pVMax");

  etaBin_ = iConfig.getParameter<int>("etaBin");
  etaMin_ = iConfig.getParameter<double>("etaMin");
  etaMax_ = iConfig.getParameter<double>("etaMax");

  ptBin_ = iConfig.getParameter<int>("ptBin");
  ptMin_ = iConfig.getParameter<double>("ptMin");
  ptMax_ = iConfig.getParameter<double>("ptMax");

  // -----------------------------------------
  // verbose_ 0: suppress printouts
  //          1: show printouts
  verbose_ = conf_.getUntrackedParameter<int>("Verbose",0);
  metFolder   = conf_.getUntrackedParameter<std::string>("metFolder");
  jetAlgo     = conf_.getUntrackedParameter<std::string>("jetAlgo");
  folderName  = conf_.getUntrackedParameter<std::string>("folderName");

  jetTests[0][0] = conf_.getUntrackedParameter<bool>("pfBarrelJetMeanTest",true);
  jetTests[0][1] = conf_.getUntrackedParameter<bool>("pfBarrelJetKSTest",false);
  jetTests[1][0] = conf_.getUntrackedParameter<bool>("pfEndcapJetMeanTest",true);
  jetTests[1][1] = conf_.getUntrackedParameter<bool>("pfEndcapJetKSTest",false);
  jetTests[2][0] = conf_.getUntrackedParameter<bool>("pfForwardJetMeanTest",true);
  jetTests[2][1] = conf_.getUntrackedParameter<bool>("pfForwardJetKSTest",false);
  jetTests[3][0] = conf_.getUntrackedParameter<bool>("caloJetMeanTest",true);
  jetTests[3][1] = conf_.getUntrackedParameter<bool>("caloJetKSTest",false);
  jetTests[4][0] = conf_.getUntrackedParameter<bool>("jptJetMeanTest",true);
  jetTests[4][1] = conf_.getUntrackedParameter<bool>("jptJetKSTest",false);

  metTests[0][0] = conf_.getUntrackedParameter<bool>("caloMETMeanTest",true);
  metTests[0][1] = conf_.getUntrackedParameter<bool>("caloMETKSTest",false);
  metTests[1][0] = conf_.getUntrackedParameter<bool>("pfMETMeanTest",true);
  metTests[1][1] = conf_.getUntrackedParameter<bool>("pfMETKSTest",false);
  metTests[2][0] = conf_.getUntrackedParameter<bool>("tcMETMeanTest",true);
  metTests[2][1] = conf_.getUntrackedParameter<bool>("tcMETKSTest",false);
 
  if (verbose_) std::cout << ">>> Constructor (DataCertificationJetMET) <<<" << std::endl;

  // -----------------------------------------
  //
}


DataCertificationJetMET::~DataCertificationJetMET()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if (verbose_) std::cout << ">>> Deconstructor (DataCertificationJetMET) <<<" << std::endl;
}


// ------------ method called right after a run ends ------------
void 
DataCertificationJetMET::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_)
{

  //put RECO vs MiniAODDir first ->first MET
  std::vector<std::string> subDirVecMET;
  std::string RunDirMET="JetMET/MET/";
  iget_.setCurrentFolder(RunDirMET);
  subDirVecMET=iget_.getSubdirs();
  bool found_METreco_dir=false;
  bool found_METminiaod_dir=false;
  //check if proper directories are inside the files
  for (int i=0; i<int(subDirVecMET.size()); i++) {
    ibook_.setCurrentFolder(subDirVecMET[i]);  
    if((subDirVecMET[i]+"/Cleaned")==(RunDirMET+inputMETLabelRECO_.label()+"/Cleaned")){
      found_METreco_dir=true;
    }
    if((subDirVecMET[i]+"/Cleaned")==(RunDirMET+inputMETLabelMiniAOD_.label()+"/Cleaned")){
      found_METminiaod_dir=true;
    }
  }
  if(found_METreco_dir && found_METminiaod_dir){
    std::string rundirMET_reco=RunDirMET+inputMETLabelRECO_.label()+"/Cleaned";
    std::string rundirMET_miniaod=RunDirMET+inputMETLabelMiniAOD_.label()+"/Cleaned";
    MonitorElement* mMET_Reco=iget_.get(rundirMET_reco+"/"+"MET");
    MonitorElement* mMEy_Reco=iget_.get(rundirMET_reco+"/"+"MEy");
    MonitorElement* mSumET_Reco=iget_.get(rundirMET_reco+"/"+"SumET");
    MonitorElement* mMETPhi_Reco=iget_.get(rundirMET_reco+"/"+"METPhi");
    MonitorElement* mMET_logx_Reco=iget_.get(rundirMET_reco+"/"+"MET_logx");
    MonitorElement* mSumET_logx_Reco=iget_.get(rundirMET_reco+"/"+"SumET_logx");
    MonitorElement* mChargedHadronEtFraction_Reco=iget_.get(rundirMET_reco+"/"+"PfChargedHadronEtFraction");
    MonitorElement* mNeutralHadronEtFraction_Reco=iget_.get(rundirMET_reco+"/"+"PfNeutralHadronEtFraction");
    MonitorElement* mPhotonEtFraction_Reco=iget_.get(rundirMET_reco+"/"+"PfPhotonEtFraction");
    MonitorElement* mHFHadronEtFraction_Reco=iget_.get(rundirMET_reco+"/"+"PfHFHadronEtFraction");
    MonitorElement* mHFEMEtFraction_Reco=iget_.get(rundirMET_reco+"/"+"PfHFEMEtFraction");
    MonitorElement* mMET_nVtx_profile_Reco=iget_.get(rundirMET_reco+"/"+"MET_profile");
    MonitorElement* mSumET_nVtx_profile_Reco=iget_.get(rundirMET_reco+"/"+"SumET_profile");
    MonitorElement* mChargedHadronEtFraction_nVtx_profile_Reco=iget_.get(rundirMET_reco+"/"+"PfChargedHadronEtFraction_profile");
    MonitorElement* mNeutralHadronEtFraction_nVtx_profile_Reco=iget_.get(rundirMET_reco+"/"+"PfNeutralHadronEtFraction_profile");
    MonitorElement* mPhotonEtFraction_nVtx_profile_Reco=iget_.get(rundirMET_reco+"/"+"PfPhotonEtFraction_profile");

    MonitorElement* mMET_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"MET");
    MonitorElement* mMEy_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"MEy");
    MonitorElement* mSumET_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"SumET");
    MonitorElement* mMETPhi_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"METPhi");
    MonitorElement* mMET_logx_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"MET_logx");
    MonitorElement* mSumET_logx_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"SumET_logx");
    MonitorElement* mChargedHadronEtFraction_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfChargedHadronEtFraction");
    MonitorElement* mNeutralHadronEtFraction_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfNeutralHadronEtFraction");
    MonitorElement* mPhotonEtFraction_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfPhotonEtFraction");
    MonitorElement* mHFHadronEtFraction_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfHFHadronEtFraction");
    MonitorElement* mHFEMEtFraction_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfHFEMEtFraction");
    MonitorElement* mMET_nVtx_profile_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"MET_profile");
    MonitorElement* mSumET_nVtx_profile_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"SumET_profile");
    MonitorElement* mChargedHadronEtFraction_nVtx_profile_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfChargedHadronEtFraction_profile");
    MonitorElement* mNeutralHadronEtFraction_nVtx_profile_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfNeutralHadronEtFraction_profile");
    MonitorElement* mPhotonEtFraction_nVtx_profile_MiniAOD=iget_.get(rundirMET_miniaod+"/"+"PfPhotonEtFraction_profile");

    ibook_.setCurrentFolder(RunDirMET+"MiniAOD_over_RECO");
    mMET_MiniAOD_over_Reco=ibook_.book1D("MET_MiniAOD_over_RECO",(TH1F*)mMET_Reco->getRootObject());
    mMEy_MiniAOD_over_Reco=ibook_.book1D("MEy_MiniAOD_over_RECO",(TH1F*)mMEy_Reco->getRootObject());
    mSumET_MiniAOD_over_Reco=ibook_.book1D("SumET_MiniAOD_over_RECO",(TH1F*)mSumET_Reco->getRootObject());
    mMETPhi_MiniAOD_over_Reco=ibook_.book1D("METPhi_MiniAOD_over_RECO",(TH1F*)mMETPhi_Reco->getRootObject());
    mMET_logx_MiniAOD_over_Reco=ibook_.book1D("MET_logx_MiniAOD_over_RECO",(TH1F*)mMET_logx_Reco->getRootObject());
    mSumET_logx_MiniAOD_over_Reco=ibook_.book1D("SumET_logx_MiniAOD_over_RECO",(TH1F*)mSumET_logx_Reco->getRootObject());
    mChargedHadronEtFraction_MiniAOD_over_Reco=ibook_.book1D("PfChargedHadronEtFraction_MiniAOD_over_RECO",(TH1F*)mChargedHadronEtFraction_Reco->getRootObject());
    mNeutralHadronEtFraction_MiniAOD_over_Reco=ibook_.book1D("PfNeutralHadronEtFraction_MiniAOD_over_RECO",(TH1F*)mNeutralHadronEtFraction_Reco->getRootObject());
    mPhotonEtFraction_MiniAOD_over_Reco=ibook_.book1D("PfPhotonEtFraction_MiniAOD_over_RECO",(TH1F*)mPhotonEtFraction_Reco->getRootObject());
    mHFHadronEtFraction_MiniAOD_over_Reco=ibook_.book1D("PfHFHadronEtFraction_MiniAOD_over_RECO",(TH1F*)mHFHadronEtFraction_Reco->getRootObject());
    mHFEMEtFraction_MiniAOD_over_Reco=ibook_.book1D("PfHFEMEtFraction_MiniAOD_over_RECO",(TH1F*)mHFEMEtFraction_Reco->getRootObject());
    //use same parameters defining X-Axis of the profiles
    mMET_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("MET_profile_MiniAOD_over_RECO","MET_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);
    mSumET_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("SumET_profile_MiniAOD_over_RECO","SumME_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);
    mChargedHadronEtFraction_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("PfChargedHadronEtFraction_profile_MiniAOD_over_RECO","PfChargedHadronEtFraction_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);
    mNeutralHadronEtFraction_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("PfNeutralHadronEtFraction_profile_MiniAOD_over_RECO","PfNeutralHadronEtFraction_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);
    mPhotonEtFraction_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("PfPhotonEtFraction_profile_MiniAOD_over_RECO","PfPhotonEtFraction_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);

    for(int i=0;i<=(mMET_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMET_Reco->getBinContent(i)!=0){
	mMET_MiniAOD_over_Reco->setBinContent(i,mMET_MiniAOD->getBinContent(i)/mMET_Reco->getBinContent(i));
      }else if(mMET_MiniAOD->getBinContent(i)!=0){
	mMET_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMEy_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMEy_Reco->getBinContent(i)!=0){
	mMEy_MiniAOD_over_Reco->setBinContent(i,mMEy_MiniAOD->getBinContent(i)/mMEy_Reco->getBinContent(i));
      }else if(mMEy_MiniAOD->getBinContent(i)!=0){
	mMEy_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mSumET_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mSumET_Reco->getBinContent(i)!=0){
	mSumET_MiniAOD_over_Reco->setBinContent(i,mSumET_MiniAOD->getBinContent(i)/mSumET_Reco->getBinContent(i));
      }else if(mSumET_MiniAOD->getBinContent(i)!=0){
	mSumET_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMETPhi_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMETPhi_Reco->getBinContent(i)!=0){
	mMETPhi_MiniAOD_over_Reco->setBinContent(i,mMETPhi_MiniAOD->getBinContent(i)/mMETPhi_Reco->getBinContent(i));
      }else if(mMETPhi_MiniAOD->getBinContent(i)!=0){
	mMETPhi_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMET_logx_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMET_logx_Reco->getBinContent(i)!=0){
	mMET_logx_MiniAOD_over_Reco->setBinContent(i,mMET_logx_MiniAOD->getBinContent(i)/mMET_logx_Reco->getBinContent(i));
      }else if(mMET_logx_MiniAOD->getBinContent(i)!=0){
	mMET_logx_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mSumET_logx_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mSumET_logx_Reco->getBinContent(i)!=0){
	mSumET_logx_MiniAOD_over_Reco->setBinContent(i,mSumET_logx_MiniAOD->getBinContent(i)/mSumET_logx_Reco->getBinContent(i));
      }else if(mSumET_logx_MiniAOD->getBinContent(i)!=0){
	mSumET_logx_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mChargedHadronEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mChargedHadronEtFraction_Reco->getBinContent(i)!=0){
	mChargedHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,mChargedHadronEtFraction_MiniAOD->getBinContent(i)/mChargedHadronEtFraction_Reco->getBinContent(i));
      }else if(mChargedHadronEtFraction_MiniAOD->getBinContent(i)!=0){
	mChargedHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    } 
    for(int i=0;i<=(mNeutralHadronEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNeutralHadronEtFraction_Reco->getBinContent(i)!=0){
	mNeutralHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,mNeutralHadronEtFraction_MiniAOD->getBinContent(i)/mNeutralHadronEtFraction_Reco->getBinContent(i));
      }else if(mNeutralHadronEtFraction_MiniAOD->getBinContent(i)!=0){
	mNeutralHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPhotonEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPhotonEtFraction_Reco->getBinContent(i)!=0){
	mPhotonEtFraction_MiniAOD_over_Reco->setBinContent(i,mPhotonEtFraction_MiniAOD->getBinContent(i)/mPhotonEtFraction_Reco->getBinContent(i));
      }else if(mPhotonEtFraction_MiniAOD->getBinContent(i)!=0){
	mPhotonEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mHFHadronEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mHFHadronEtFraction_Reco->getBinContent(i)!=0){
	mHFHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,mHFHadronEtFraction_MiniAOD->getBinContent(i)/mHFHadronEtFraction_Reco->getBinContent(i));
      }else if(mHFHadronEtFraction_MiniAOD->getBinContent(i)!=0){
	mHFHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mHFEMEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mHFEMEtFraction_Reco->getBinContent(i)!=0){
	mHFEMEtFraction_MiniAOD_over_Reco->setBinContent(i,mHFEMEtFraction_MiniAOD->getBinContent(i)/mHFEMEtFraction_Reco->getBinContent(i));
      }else if(mHFEMEtFraction_MiniAOD->getBinContent(i)!=0){
	mHFEMEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMET_nVtx_profile_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMET_nVtx_profile_Reco->getBinContent(i)!=0){
	mMET_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,mMET_nVtx_profile_MiniAOD->getBinContent(i)/mMET_nVtx_profile_Reco->getBinContent(i));
      }else if(mMET_nVtx_profile_MiniAOD->getBinContent(i)!=0){
	mMET_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mSumET_nVtx_profile_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mSumET_nVtx_profile_Reco->getBinContent(i)!=0){
	mSumET_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,mSumET_nVtx_profile_MiniAOD->getBinContent(i)/mSumET_nVtx_profile_Reco->getBinContent(i));
      }else if(mSumET_nVtx_profile_MiniAOD->getBinContent(i)!=0){
	mSumET_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mChargedHadronEtFraction_nVtx_profile_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mChargedHadronEtFraction_nVtx_profile_Reco->getBinContent(i)!=0){
	mChargedHadronEtFraction_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,mChargedHadronEtFraction_nVtx_profile_MiniAOD->getBinContent(i)/mChargedHadronEtFraction_nVtx_profile_Reco->getBinContent(i));
      }else if(mChargedHadronEtFraction_nVtx_profile_MiniAOD->getBinContent(i)!=0){
	mChargedHadronEtFraction_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mNeutralHadronEtFraction_nVtx_profile_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNeutralHadronEtFraction_nVtx_profile_Reco->getBinContent(i)!=0){
	mNeutralHadronEtFraction_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,mNeutralHadronEtFraction_nVtx_profile_MiniAOD->getBinContent(i)/mNeutralHadronEtFraction_nVtx_profile_Reco->getBinContent(i));
      }else if(mNeutralHadronEtFraction_nVtx_profile_MiniAOD->getBinContent(i)!=0){
	mNeutralHadronEtFraction_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPhotonEtFraction_nVtx_profile_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPhotonEtFraction_nVtx_profile_Reco->getBinContent(i)!=0){
	mPhotonEtFraction_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,mPhotonEtFraction_nVtx_profile_MiniAOD->getBinContent(i)/mPhotonEtFraction_nVtx_profile_Reco->getBinContent(i));
      }else if(mPhotonEtFraction_nVtx_profile_MiniAOD->getBinContent(i)!=0){
	mPhotonEtFraction_nVtx_profile_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
  }



  //put RECO vs MiniAODDir first ->second Jets
  std::vector<std::string> subDirVecJet;
  //go only for cleaned directory
  std::string RunDirJet="JetMET/Jet/";
  iget_.setCurrentFolder(RunDirJet);
  subDirVecJet=iget_.getSubdirs();
  bool found_Jetreco_dir=false;
  bool found_Jetminiaod_dir=false;
  for (int i=0; i<int(subDirVecJet.size()); i++) {
    ibook_.setCurrentFolder(subDirVecJet[i]);  
    if(subDirVecJet[i]==(RunDirJet+"Cleaned"+inputJetLabelRECO_.label())){
      found_Jetreco_dir=true;
    }
    if(subDirVecJet[i]==(RunDirJet+"Cleaned"+inputJetLabelMiniAOD_.label())){
      found_Jetminiaod_dir=true;
    }
  }
  if(found_Jetreco_dir && found_Jetminiaod_dir){
    std::string rundirJet_reco=RunDirJet+"Cleaned"+inputJetLabelRECO_.label();
    std::string rundirJet_miniaod=RunDirJet+"Cleaned"+inputJetLabelMiniAOD_.label();

    MonitorElement* mPt_Reco=iget_.get(rundirJet_reco+"/"+"Pt");
    MonitorElement* mEta_Reco=iget_.get(rundirJet_reco+"/"+"Eta");
    MonitorElement* mPhi_Reco=iget_.get(rundirJet_reco+"/"+"Phi");
    MonitorElement* mNjets_Reco=iget_.get(rundirJet_reco+"/"+"NJets");
    MonitorElement* mPt_uncor_Reco=iget_.get(rundirJet_reco+"/"+"Pt_uncor");
    MonitorElement* mEta_uncor_Reco=iget_.get(rundirJet_reco+"/"+"Eta_uncor");
    MonitorElement* mPhi_uncor_Reco=iget_.get(rundirJet_reco+"/"+"Phi_uncor");
    MonitorElement* mJetEnergyCorr_Reco=iget_.get(rundirJet_reco+"/"+"JetEnergyCorr");
    MonitorElement* mJetEnergyCorrVSeta_Reco=iget_.get(rundirJet_reco+"/"+"JetEnergyCorrVSEta");
    MonitorElement* mDPhi_Reco=iget_.get(rundirJet_reco+"/"+"DPhi");
    MonitorElement* mLooseJIDPassFractionVSeta_Reco=iget_.get(rundirJet_reco+"/"+"JetIDPassFractionVSeta");
    MonitorElement* mPt_Barrel_Reco=iget_.get(rundirJet_reco+"/"+"Pt_Barrel");
    MonitorElement* mPt_EndCap_Reco=iget_.get(rundirJet_reco+"/"+"Pt_EndCap");
    MonitorElement* mPt_Forward_Reco=iget_.get(rundirJet_reco+"/"+"Pt_Forward");
    MonitorElement* mMVAPUJIDDiscriminant_lowPt_Barrel_Reco=iget_.get(rundirJet_reco+"/"+"MVAPUJIDDiscriminant_lowPt_Barrel");
    MonitorElement* mMVAPUJIDDiscriminant_lowPt_EndCap_Reco=iget_.get(rundirJet_reco+"/"+"MVAPUJIDDiscriminant_lowPt_EndCap");
    MonitorElement* mMVAPUJIDDiscriminant_lowPt_Forward_Reco=iget_.get(rundirJet_reco+"/"+"MVAPUJIDDiscriminant_lowPt_Forward");
    MonitorElement* mMVAPUJIDDiscriminant_mediumPt_EndCap_Reco=iget_.get(rundirJet_reco+"/"+"MVAPUJIDDiscriminant_mediumPt_EndCap");
    MonitorElement* mMVAPUJIDDiscriminant_highPt_Barrel_Reco=iget_.get(rundirJet_reco+"/"+"MVAPUJIDDiscriminant_highPt_Barrel");
    MonitorElement* mCHFracVSpT_Barrel_Reco=iget_.get(rundirJet_reco+"/"+"CHFracVSpT_Barrel");
    MonitorElement* mNHFracVSpT_EndCap_Reco=iget_.get(rundirJet_reco+"/"+"NHFracVSpT_EndCap");
    MonitorElement* mPhFracVSpT_Barrel_Reco=iget_.get(rundirJet_reco+"/"+"PhFracVSpT_Barrel");
    MonitorElement* mHFHFracVSpT_Forward_Reco=iget_.get(rundirJet_reco+"/"+"HFHFracVSpT_Forward");
    MonitorElement* mHFEFracVSpT_Forward_Reco=iget_.get(rundirJet_reco+"/"+"HFEFracVSpT_Forward");
    MonitorElement* mCHFrac_Reco=iget_.get(rundirJet_reco+"/DiJet/"+"CHFrac");
    MonitorElement* mNHFrac_Reco=iget_.get(rundirJet_reco+"/DiJet/"+"NHFrac");
    MonitorElement* mPhFrac_Reco=iget_.get(rundirJet_reco+"/DiJet/"+"PhFrac");
    MonitorElement* mChargedMultiplicity_Reco=iget_.get(rundirJet_reco+"/DiJet/"+"ChargedMultiplicity");
    MonitorElement* mNeutralMultiplicity_Reco=iget_.get(rundirJet_reco+"/DiJet/"+"NeutralMultiplicity");
    MonitorElement* mMuonMultiplicity_Reco=iget_.get(rundirJet_reco+"/DiJet/"+"MuonMultiplicity");
    MonitorElement* mNeutralFraction_Reco=iget_.get(rundirJet_reco+"/DiJet/"+"NeutralConstituentsFraction");    
    MonitorElement* mPt_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Pt");
    MonitorElement* mEta_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Eta");
    MonitorElement* mPhi_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Phi");
    MonitorElement* mNjets_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"NJets");
    MonitorElement* mPt_uncor_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Pt_uncor");
    MonitorElement* mEta_uncor_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Eta_uncor");
    MonitorElement* mPhi_uncor_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Phi_uncor");
    MonitorElement* mJetEnergyCorr_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"JetEnergyCorr");
    MonitorElement* mJetEnergyCorrVSeta_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"JetEnergyCorrVSEta");
    MonitorElement* mDPhi_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"DPhi");
    MonitorElement* mLooseJIDPassFractionVSeta_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"JetIDPassFractionVSeta");
    MonitorElement* mPt_Barrel_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Pt_Barrel");
    MonitorElement* mPt_EndCap_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Pt_EndCap");
    MonitorElement* mPt_Forward_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"Pt_Forward");
    MonitorElement* mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"MVAPUJIDDiscriminant_lowPt_Barrel");
    MonitorElement* mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"MVAPUJIDDiscriminant_lowPt_EndCap");
    MonitorElement* mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"MVAPUJIDDiscriminant_lowPt_Forward");
    MonitorElement* mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"MVAPUJIDDiscriminant_mediumPt_EndCap");
    MonitorElement* mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"MVAPUJIDDiscriminant_highPt_Barrel");
    MonitorElement* mCHFracVSpT_Barrel_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"CHFracVSpT_Barrel");
    MonitorElement* mNHFracVSpT_EndCap_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"NHFracVSpT_EndCap");
    MonitorElement* mPhFracVSpT_Barrel_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"PhFracVSpT_Barrel");
    MonitorElement* mHFHFracVSpT_Forward_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"HFHFracVSpT_Forward");
    MonitorElement* mHFEFracVSpT_Forward_MiniAOD=iget_.get(rundirJet_miniaod+"/"+"HFEFracVSpT_Forward");
    MonitorElement* mCHFrac_MiniAOD=iget_.get(rundirJet_miniaod+"/DiJet/"+"CHFrac");
    MonitorElement* mNHFrac_MiniAOD=iget_.get(rundirJet_miniaod+"/DiJet/"+"NHFrac");
    MonitorElement* mPhFrac_MiniAOD=iget_.get(rundirJet_miniaod+"/DiJet/"+"PhFrac");
    MonitorElement* mChargedMultiplicity_MiniAOD=iget_.get(rundirJet_miniaod+"/DiJet/"+"ChargedMultiplicity");
    MonitorElement* mNeutralMultiplicity_MiniAOD=iget_.get(rundirJet_miniaod+"/DiJet/"+"NeutralMultiplicity");
    MonitorElement* mMuonMultiplicity_MiniAOD=iget_.get(rundirJet_miniaod+"/DiJet/"+"MuonMultiplicity");
    MonitorElement* mNeutralFraction_MiniAOD=iget_.get(rundirJet_miniaod+"/DiJet/"+"NeutralConstituentsFraction");    
    ibook_.setCurrentFolder(RunDirJet+"MiniAOD_over_RECO");
    mPt_MiniAOD_over_Reco=ibook_.book1D("Pt_MiniAOD_over_RECO",(TH1F*)mPt_Reco->getRootObject());
    mEta_MiniAOD_over_Reco=ibook_.book1D("Eta_MiniAOD_over_RECO",(TH1F*)mEta_Reco->getRootObject());
    mPhi_MiniAOD_over_Reco=ibook_.book1D("Phi_MiniAOD_over_RECO",(TH1F*)mPhi_Reco->getRootObject());
    mNjets_MiniAOD_over_Reco=ibook_.book1D("NJets_MiniAOD_over_RECO",(TH1F*)mNjets_Reco->getRootObject());
    mPt_uncor_MiniAOD_over_Reco=ibook_.book1D("Pt_uncor_MiniAOD_over_RECO",(TH1F*)mPt_uncor_Reco->getRootObject());
    mEta_uncor_MiniAOD_over_Reco=ibook_.book1D("Eta_uncor_MiniAOD_over_RECO",(TH1F*)mEta_uncor_Reco->getRootObject());
    mPhi_uncor_MiniAOD_over_Reco=ibook_.book1D("Phi_uncor_MiniAOD_over_RECO",(TH1F*)mPhi_uncor_Reco->getRootObject());
    mJetEnergyCorr_MiniAOD_over_Reco=ibook_.book1D("JetEnergyCorr_MiniAOD_over_RECO",(TH1F*)mJetEnergyCorr_Reco->getRootObject());
    mJetEnergyCorrVSeta_MiniAOD_over_Reco=ibook_.book1D("JetEnergyCorrVSEta_MiniAOD_over_RECO",  "jet energy correction factor VS eta", etaBin_, etaMin_,etaMax_);
    mDPhi_MiniAOD_over_Reco=ibook_.book1D("DPhi_MiniAOD_over_RECO",(TH1F*)mDPhi_Reco->getRootObject());
    mLooseJIDPassFractionVSeta_MiniAOD_over_Reco=ibook_.book1D("JetIDPassFractionVSeta_MiniAOD_over_RECO","JetIDPassFractionVSeta", etaBin_, etaMin_,etaMax_);
    mPt_Barrel_MiniAOD_over_Reco=ibook_.book1D("Pt_Barrel_MiniAOD_over_RECO",(TH1F*)mPt_Barrel_Reco->getRootObject());
    mPt_EndCap_MiniAOD_over_Reco=ibook_.book1D("Pt_EndCap_MiniAOD_over_RECO",(TH1F*)mPt_EndCap_Reco->getRootObject());
    mPt_Forward_MiniAOD_over_Reco=ibook_.book1D("Pt_Forward_MiniAOD_over_RECO",(TH1F*)mPt_Forward_Reco->getRootObject());
    mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD_over_Reco=ibook_.book1D("MVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD_over_RECO",(TH1F*)mMVAPUJIDDiscriminant_lowPt_Barrel_Reco->getRootObject());
    mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD_over_Reco=ibook_.book1D("MVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD_over_RECO",(TH1F*)mMVAPUJIDDiscriminant_lowPt_EndCap_Reco->getRootObject());
    mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD_over_Reco=ibook_.book1D("MVAPUJIDDiscriminant_lowPt_Forward_MiniAOD_over_RECO",(TH1F*)mMVAPUJIDDiscriminant_lowPt_Forward_Reco->getRootObject());
    mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD_over_Reco=ibook_.book1D("MVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD_over_RECO",(TH1F*)mMVAPUJIDDiscriminant_mediumPt_EndCap_Reco->getRootObject());
    mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD_over_Reco=ibook_.book1D("MVAPUJIDDiscriminant_highPt_Barrel_MiniAOD_over_RECO",(TH1F*)mMVAPUJIDDiscriminant_highPt_Barrel_Reco->getRootObject());
    mCHFracVSpT_Barrel_MiniAOD_over_Reco=ibook_.book1D("CHFracVSpT_Barrel_MiniAOD_over_RECO","CHFracVSpT_Barrel", ptBin_, ptMin_,ptMax_);
    mNHFracVSpT_EndCap_MiniAOD_over_Reco=ibook_.book1D("NHFracVSpT_EndCap_MiniAOD_over_RECO","NHFracVSpT_EndCap", ptBin_, ptMin_,ptMax_);
    mPhFracVSpT_Barrel_MiniAOD_over_Reco=ibook_.book1D("PhFracVSpT_Barrel_MiniAOD_over_RECO","PhFracVSpT_Barrel", ptBin_, ptMin_,ptMax_);
    mHFHFracVSpT_Forward_MiniAOD_over_Reco=ibook_.book1D("HFHFracVSpT_Forward_MiniAOD_over_RECO","HFHFracVSpT_Forward", ptBin_, ptMin_,ptMax_);
    mHFEFracVSpT_Forward_MiniAOD_over_Reco=ibook_.book1D("HFEFracVSpT_Forward_MiniAOD_over_RECO","HFEFracVSpT_Forward", ptBin_, ptMin_,ptMax_);
    mCHFrac_MiniAOD_over_Reco=ibook_.book1D("CHFrac_MiniAOD_over_RECO",(TH1F*)mCHFrac_Reco->getRootObject());
    mNHFrac_MiniAOD_over_Reco=ibook_.book1D("NHFrac_MiniAOD_over_RECO",(TH1F*)mNHFrac_Reco->getRootObject());
    mPhFrac_MiniAOD_over_Reco=ibook_.book1D("PhFrac_MiniAOD_over_RECO",(TH1F*)mPhFrac_Reco->getRootObject());
    mChargedMultiplicity_MiniAOD_over_Reco=ibook_.book1D("ChargedMultiplicity_MiniAOD_over_RECO",(TH1F*)mChargedMultiplicity_Reco->getRootObject());
    mNeutralMultiplicity_MiniAOD_over_Reco=ibook_.book1D("NeutralMultiplicity_MiniAOD_over_RECO",(TH1F*)mNeutralMultiplicity_Reco->getRootObject());
    mMuonMultiplicity_MiniAOD_over_Reco=ibook_.book1D("MuonMultiplicity_MiniAOD_over_RECO",(TH1F*)mMuonMultiplicity_Reco->getRootObject());
    mNeutralFraction_MiniAOD_over_Reco=ibook_.book1D("NeutralConstituentsFraction_MiniAOD_over_RECO",(TH1F*)mNeutralFraction_Reco->getRootObject());
    for(int i=0;i<=(mPt_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPt_Reco->getBinContent(i)!=0){
	mPt_MiniAOD_over_Reco->setBinContent(i,mPt_MiniAOD->getBinContent(i)/mPt_Reco->getBinContent(i));
      }else if(mPt_MiniAOD->getBinContent(i)!=0){
	mPt_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mEta_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mEta_Reco->getBinContent(i)!=0){
	mEta_MiniAOD_over_Reco->setBinContent(i,mEta_MiniAOD->getBinContent(i)/mEta_Reco->getBinContent(i));
      }else if(mEta_MiniAOD->getBinContent(i)!=0){
	mEta_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPhi_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPhi_Reco->getBinContent(i)!=0){
	mPhi_MiniAOD_over_Reco->setBinContent(i,mPhi_MiniAOD->getBinContent(i)/mPhi_Reco->getBinContent(i));
      }else if(mPhi_MiniAOD->getBinContent(i)!=0){
	mPhi_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mNjets_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNjets_Reco->getBinContent(i)!=0){
	mNjets_MiniAOD_over_Reco->setBinContent(i,mNjets_MiniAOD->getBinContent(i)/mNjets_Reco->getBinContent(i));
      }else if(mNjets_MiniAOD->getBinContent(i)!=0){
	mNjets_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPt_uncor_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPt_uncor_Reco->getBinContent(i)!=0){
	mPt_uncor_MiniAOD_over_Reco->setBinContent(i,mPt_uncor_MiniAOD->getBinContent(i)/mPt_uncor_Reco->getBinContent(i));
      }else if(mPt_uncor_MiniAOD->getBinContent(i)!=0){
	mPt_uncor_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mEta_uncor_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mEta_uncor_Reco->getBinContent(i)!=0){
	mEta_uncor_MiniAOD_over_Reco->setBinContent(i,mEta_uncor_MiniAOD->getBinContent(i)/mEta_uncor_Reco->getBinContent(i));
      }else if(mEta_uncor_MiniAOD->getBinContent(i)!=0){
	mEta_uncor_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPhi_uncor_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPhi_uncor_Reco->getBinContent(i)!=0){
	mPhi_uncor_MiniAOD_over_Reco->setBinContent(i,mPhi_uncor_MiniAOD->getBinContent(i)/mPhi_uncor_Reco->getBinContent(i));
      }else if(mPhi_uncor_MiniAOD->getBinContent(i)!=0){
	mPhi_uncor_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mJetEnergyCorr_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mJetEnergyCorr_Reco->getBinContent(i)!=0){
	mJetEnergyCorr_MiniAOD_over_Reco->setBinContent(i,mJetEnergyCorr_MiniAOD->getBinContent(i)/mJetEnergyCorr_Reco->getBinContent(i));
      }else if(mJetEnergyCorr_MiniAOD->getBinContent(i)!=0){
	mJetEnergyCorr_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mJetEnergyCorrVSeta_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mJetEnergyCorrVSeta_Reco->getBinContent(i)!=0){
	mJetEnergyCorrVSeta_MiniAOD_over_Reco->setBinContent(i,mJetEnergyCorrVSeta_MiniAOD->getBinContent(i)/mJetEnergyCorrVSeta_Reco->getBinContent(i));
      }else if(mJetEnergyCorrVSeta_MiniAOD->getBinContent(i)!=0){
	mJetEnergyCorrVSeta_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mDPhi_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mDPhi_Reco->getBinContent(i)!=0){
	mDPhi_MiniAOD_over_Reco->setBinContent(i,mDPhi_MiniAOD->getBinContent(i)/mDPhi_Reco->getBinContent(i));
      }else if(mDPhi_MiniAOD->getBinContent(i)!=0){
	mDPhi_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mLooseJIDPassFractionVSeta_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mLooseJIDPassFractionVSeta_Reco->getBinContent(i)!=0){
	mLooseJIDPassFractionVSeta_MiniAOD_over_Reco->setBinContent(i,mLooseJIDPassFractionVSeta_MiniAOD->getBinContent(i)/mLooseJIDPassFractionVSeta_Reco->getBinContent(i));
      }else if(mLooseJIDPassFractionVSeta_MiniAOD_over_Reco->getBinContent(i)!=0){
	mLooseJIDPassFractionVSeta_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPt_Barrel_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPt_Barrel_Reco->getBinContent(i)!=0){
	mPt_Barrel_MiniAOD_over_Reco->setBinContent(i,mPt_Barrel_MiniAOD->getBinContent(i)/mPt_Barrel_Reco->getBinContent(i));
      }else if(mPt_Barrel_MiniAOD->getBinContent(i)!=0){
	mPt_Barrel_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPt_EndCap_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPt_EndCap_Reco->getBinContent(i)!=0){
	mPt_EndCap_MiniAOD_over_Reco->setBinContent(i,mPt_EndCap_MiniAOD->getBinContent(i)/mPt_EndCap_Reco->getBinContent(i));
      }else if(mPt_EndCap_MiniAOD->getBinContent(i)!=0){
	mPt_EndCap_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPt_Forward_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPt_Forward_Reco->getBinContent(i)!=0){
	mPt_Forward_MiniAOD_over_Reco->setBinContent(i,mPt_Forward_MiniAOD->getBinContent(i)/mPt_Forward_Reco->getBinContent(i));
      }else if(mPt_Forward_MiniAOD->getBinContent(i)!=0){
	mPt_Forward_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMVAPUJIDDiscriminant_lowPt_Barrel_Reco->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD_over_Reco->setBinContent(i,mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD->getBinContent(i)/mMVAPUJIDDiscriminant_lowPt_Barrel_Reco->getBinContent(i));
      }else if(mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMVAPUJIDDiscriminant_lowPt_EndCap_Reco->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD_over_Reco->setBinContent(i,mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD->getBinContent(i)/mMVAPUJIDDiscriminant_lowPt_EndCap_Reco->getBinContent(i));
      }else if(mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMVAPUJIDDiscriminant_lowPt_Forward_Reco->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD_over_Reco->setBinContent(i,mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD->getBinContent(i)/mMVAPUJIDDiscriminant_lowPt_Forward_Reco->getBinContent(i));
      }else if(mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMVAPUJIDDiscriminant_mediumPt_EndCap_Reco->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD_over_Reco->setBinContent(i,mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD->getBinContent(i)/mMVAPUJIDDiscriminant_mediumPt_EndCap_Reco->getBinContent(i));
      }else if(mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMVAPUJIDDiscriminant_highPt_Barrel_Reco->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD_over_Reco->setBinContent(i,mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD->getBinContent(i)/mMVAPUJIDDiscriminant_highPt_Barrel_Reco->getBinContent(i));
      }else if(mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD->getBinContent(i)!=0){
	mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mCHFracVSpT_Barrel_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mCHFracVSpT_Barrel_Reco->getBinContent(i)!=0){
	mCHFracVSpT_Barrel_MiniAOD_over_Reco->setBinContent(i,mCHFracVSpT_Barrel_MiniAOD->getBinContent(i)/mCHFracVSpT_Barrel_Reco->getBinContent(i));
      }else if(mCHFracVSpT_Barrel_MiniAOD->getBinContent(i)!=0){
	mCHFracVSpT_Barrel_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mNHFracVSpT_EndCap_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNHFracVSpT_EndCap_Reco->getBinContent(i)!=0){
	mNHFracVSpT_EndCap_MiniAOD_over_Reco->setBinContent(i,mNHFracVSpT_EndCap_MiniAOD->getBinContent(i)/mNHFracVSpT_EndCap_Reco->getBinContent(i));
      }else if(mNHFracVSpT_EndCap_MiniAOD->getBinContent(i)!=0){
	mNHFracVSpT_EndCap_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPhFracVSpT_Barrel_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPhFracVSpT_Barrel_Reco->getBinContent(i)!=0){
	mPhFracVSpT_Barrel_MiniAOD_over_Reco->setBinContent(i,mPhFracVSpT_Barrel_MiniAOD->getBinContent(i)/mPhFracVSpT_Barrel_Reco->getBinContent(i));
      }else if(mPhFracVSpT_Barrel_MiniAOD->getBinContent(i)!=0){
	mPhFracVSpT_Barrel_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    } 
    for(int i=0;i<=(mHFHFracVSpT_Forward_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mHFHFracVSpT_Forward_Reco->getBinContent(i)!=0){
	mHFHFracVSpT_Forward_MiniAOD_over_Reco->setBinContent(i,mHFHFracVSpT_Forward_MiniAOD->getBinContent(i)/mHFHFracVSpT_Forward_Reco->getBinContent(i));
      }else if(mHFHFracVSpT_Forward_MiniAOD->getBinContent(i)!=0){
	mHFHFracVSpT_Forward_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mHFEFracVSpT_Forward_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mHFEFracVSpT_Forward_Reco->getBinContent(i)!=0){
	mHFEFracVSpT_Forward_MiniAOD_over_Reco->setBinContent(i,mHFEFracVSpT_Forward_MiniAOD->getBinContent(i)/mHFEFracVSpT_Forward_Reco->getBinContent(i));
      }else if(mHFEFracVSpT_Forward_MiniAOD->getBinContent(i)!=0){
	mHFEFracVSpT_Forward_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mCHFrac_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mCHFrac_Reco->getBinContent(i)!=0){
	mCHFrac_MiniAOD_over_Reco->setBinContent(i,mCHFrac_MiniAOD->getBinContent(i)/mCHFrac_Reco->getBinContent(i));
      }else if(mCHFrac_MiniAOD->getBinContent(i)!=0){
	mCHFrac_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mNHFrac_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNHFrac_Reco->getBinContent(i)!=0){
	mNHFrac_MiniAOD_over_Reco->setBinContent(i,mNHFrac_MiniAOD->getBinContent(i)/mNHFrac_Reco->getBinContent(i));
      }else if(mNHFrac_MiniAOD->getBinContent(i)!=0){
	mNHFrac_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPhFrac_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPhFrac_Reco->getBinContent(i)!=0){
	mPhFrac_MiniAOD_over_Reco->setBinContent(i,mPhFrac_MiniAOD->getBinContent(i)/mPhFrac_Reco->getBinContent(i));
      }else if(mPhFrac_MiniAOD->getBinContent(i)!=0){
	mPhFrac_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mChargedMultiplicity_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mChargedMultiplicity_Reco->getBinContent(i)!=0){
	mChargedMultiplicity_MiniAOD_over_Reco->setBinContent(i,mChargedMultiplicity_MiniAOD->getBinContent(i)/mChargedMultiplicity_Reco->getBinContent(i));
      }else if(mChargedMultiplicity_MiniAOD->getBinContent(i)!=0){
	mChargedMultiplicity_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mNeutralMultiplicity_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNeutralMultiplicity_Reco->getBinContent(i)!=0){
	mNeutralMultiplicity_MiniAOD_over_Reco->setBinContent(i,mNeutralMultiplicity_MiniAOD->getBinContent(i)/mNeutralMultiplicity_Reco->getBinContent(i));
      }else if(mNeutralMultiplicity_MiniAOD->getBinContent(i)!=0){
	mNeutralMultiplicity_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMuonMultiplicity_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMuonMultiplicity_Reco->getBinContent(i)!=0){
	mMuonMultiplicity_MiniAOD_over_Reco->setBinContent(i,mMuonMultiplicity_MiniAOD->getBinContent(i)/mMuonMultiplicity_Reco->getBinContent(i));
      }else if(mMuonMultiplicity_MiniAOD->getBinContent(i)!=0){
	mMuonMultiplicity_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mNeutralFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNeutralFraction_Reco->getBinContent(i)!=0){
	mNeutralFraction_MiniAOD_over_Reco->setBinContent(i,mNeutralFraction_MiniAOD->getBinContent(i)/mNeutralFraction_Reco->getBinContent(i));
      }else if(mNeutralFraction_MiniAOD->getBinContent(i)!=0){
	mNeutralFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
  }

  if (verbose_) std::cout << ">>> EndRun (DataCertificationJetMET) <<<" << std::endl;

  std::vector<std::string> subDirVec;
  std::string RunDir;


  if (verbose_) std::cout << "InMemory_           = " << InMemory_    << std::endl;

  ibook_.setCurrentFolder(folderName);  
  reportSummary = ibook_.bookFloat("reportSummary");
  CertificationSummary = ibook_.bookFloat("CertificationSummary");
  
  reportSummaryMap = ibook_.book2D("reportSummaryMap","reportSummaryMap",3,0,3,5,0,5);
  CertificationSummaryMap = ibook_.book2D("CertificationSummaryMap","CertificationSummaryMap",3,0,3,5,0,5);


  reportSummary = iget_.get(folderName+"/"+"reportSummary");
  CertificationSummary = iget_.get(folderName+"/"+"CertificationSummary");
  reportSummaryMap = iget_.get(folderName+"/"+"reportSummaryMap");
  CertificationSummaryMap = iget_.get(folderName+"/"+"CertificationSummaryMap");


  
  if(reportSummaryMap && reportSummaryMap->getRootObject()){ 
    reportSummaryMap->getTH2F()->SetStats(kFALSE);
    reportSummaryMap->getTH2F()->SetOption("colz");
    reportSummaryMap->setBinLabel(1,"CaloTower");
    reportSummaryMap->setBinLabel(2,"MET");
    reportSummaryMap->setBinLabel(3,"Jet");
  }
  if(CertificationSummaryMap && CertificationSummaryMap->getRootObject()){ 
    CertificationSummaryMap->getTH2F()->SetStats(kFALSE);
    CertificationSummaryMap->getTH2F()->SetOption("colz");
    CertificationSummaryMap->setBinLabel(1,"CaloTower");
    CertificationSummaryMap->setBinLabel(2,"MET");
    CertificationSummaryMap->setBinLabel(3,"Jet");
  }

  reportSummary->Fill(1.);
  CertificationSummary->Fill(1.);

  if (RunDir=="Reference") RunDir="";
  if (verbose_) std::cout << RunDir << std::endl;
  ibook_.setCurrentFolder("JetMET/EventInfo/CertificationSummaryContents/");    


  std::string refHistoName;
  std::string newHistoName;
  
  //-----------------------------
  // Jet DQM Data Certification
  //-----------------------------
  //we have 4 types anymore: PF (barrel,endcap,forward) and calojets
  MonitorElement *meJetPt[4];
  MonitorElement *meJetEta[4];
  MonitorElement *meJetPhi[4];
  MonitorElement *meJetEMFrac[4];
  MonitorElement *meJetConstituents[4];
  RunDir = "";
  if (RunDir == "") newHistoName = "JetMET/Jet/";
  else              newHistoName = RunDir+"/JetMET/Runsummary/Jet/";
  std::string cleaningdir = "";
    cleaningdir = "Cleaned";
 
 //Jet Phi histos
  meJetPhi[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_Barrel");
  meJetPhi[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_EndCap");
  meJetPhi[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Phi_Forward");
  meJetPhi[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Phi");
  //meJetPhi[4] = iget_.get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/Phi");

  //Jet Eta histos
  meJetEta[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Eta");
  meJetEta[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Eta");
  meJetEta[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EtaFirst");
  meJetEta[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Eta");
  //meJetEta[4] = iget_.get(newHistoName+cleaningdir+"JetPlusTrackZSPCorJetAntiKt5/Eta");

  //Jet Pt histos
  meJetPt[0]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_Barrel");
  meJetPt[1]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_EndCap");
  meJetPt[2]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Pt_Forward");
  meJetPt[3]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Pt_2");

  ////Jet Constituents histos
  meJetConstituents[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Barrel");
  meJetConstituents[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_EndCap");
  meJetConstituents[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Forward");
  meJetConstituents[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Constituents");
  //
  ////Jet EMFrac histos
  meJetEMFrac[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Barrel");
  meJetEMFrac[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_EndCap");
  meJetEMFrac[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Forward");
  meJetEMFrac[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/EFrac");

				   
  //------------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for Jets
  //--- Tests for Calo Barrel, EndCap and Forward, as well as PF and JPT jets
  //--- For Calo and PF jets:
  //--- Look at mean of Constituents, EM Frac and Pt
  //--- Look at Kolmogorov result for Eta, Phi, and Pt
  //--- For JPT jets:
  //--- Look at mean of Pt, AllPionsTrackNHits?, nTracks, 
  //--- Look at Kolmogorov result for Eta, Phi, and Pt
  //------------------------------------------------------------------------------


  // Four types of jets {AK5 Barrel, AK5 EndCap, AK5 Forward, PF}, removed JPT which is 5th type of jets
  //----------------------------------------------------------------------------
  // Kolmogorov (KS) tests
  const QReport* QReport_JetEta[4] = {0};
  const QReport* QReport_JetPhi[4] = {0};
  // Mean and KS tests for Calo and PF jets
  const QReport* QReport_JetConstituents[4][2] = {{0}};
  const QReport* QReport_JetEFrac[4][2]        = {{0}};
  const QReport* QReport_JetPt[4][2]           = {{0}};

  // Mean and KS tests for JPT jets
  //const QReport* QReport_JetNTracks[2] = {0, 0};
  float qr_Jet_Eta[4]     = {-1};
  float qr_Jet_Phi[4]     = {-1};
  float dc_Jet[4]         = {-1};

  float qr_Jet_Constituents[4][2] = {{-1}};
  float qr_Jet_EFrac[4][2]        = {{-1}};
  float qr_Jet_Pt[4][2]           = {{-1}};


  // Loop
  //----------------------------------------------------------------------------
  for (int jtyp=0; jtyp<4; ++jtyp) {
    // Mean test results

    if (meJetConstituents[jtyp] && meJetConstituents[jtyp]->getRootObject() ) {
      QReport_JetConstituents[jtyp][0] = meJetConstituents[jtyp]->getQReport("meanJetConstituentsTest");
      QReport_JetConstituents[jtyp][1] = meJetConstituents[jtyp]->getQReport("KolmogorovTest");
    }
    if (meJetEMFrac[jtyp]&& meJetEMFrac[jtyp]->getRootObject() ) {
      QReport_JetEFrac[jtyp][0]        = meJetEMFrac[jtyp]->getQReport("meanEMFractionTest");
      QReport_JetEFrac[jtyp][1]        = meJetEMFrac[jtyp]->getQReport("KolmogorovTest");
    }
    if (meJetPt[jtyp] && meJetPt[jtyp]->getRootObject() ) {
      QReport_JetPt[jtyp][0] = meJetPt[jtyp]->getQReport("meanJetPtTest");
      QReport_JetPt[jtyp][1] = meJetPt[jtyp]->getQReport("KolmogorovTest");
    }
    if (meJetPhi[jtyp] && meJetPhi[jtyp]->getRootObject()){
      QReport_JetPhi[jtyp]   = meJetPhi[jtyp]->getQReport("KolmogorovTest");
    }
    if (meJetEta[jtyp] && meJetEta[jtyp]->getRootObject()){
      QReport_JetEta[jtyp]   = meJetEta[jtyp]->getQReport("KolmogorovTest");
    }
    
    //Jet Pt test
    if (QReport_JetPt[jtyp][0]){
      //std::cout<<"jet type test pt "<<jtyp<<"/"<<QReport_JetPt[jtyp][0]->getStatus()<<std::endl;
      if (QReport_JetPt[jtyp][0]->getStatus()==100 ||
	  QReport_JetPt[jtyp][0]->getStatus()==200)
	qr_Jet_Pt[jtyp][0] = 1;
      else if (QReport_JetPt[jtyp][0]->getStatus()==300)
	qr_Jet_Pt[jtyp][0] = 0;
      else 
	qr_Jet_Pt[jtyp][0] = -1;
    }
    else{ qr_Jet_Pt[jtyp][0] = -2;
      //std::cout<<"qreport is REALLY NULL type test pt "<<jtyp<<" 0 "<<std::endl;
    }
    if (QReport_JetPt[jtyp][1]){
      if (QReport_JetPt[jtyp][1]->getStatus()==100 ||
	  QReport_JetPt[jtyp][1]->getStatus()==200) 
	qr_Jet_Pt[jtyp][1] = 1;
      else if (QReport_JetPt[jtyp][1]->getStatus()==300) 
	qr_Jet_Pt[jtyp][1] = 0;
      else
	qr_Jet_Pt[jtyp][1] = -1;
    }
    else{ qr_Jet_Pt[jtyp][1] = -2;
    }
    
    //Jet Phi test
    if (QReport_JetPhi[jtyp]){
      if (QReport_JetPhi[jtyp]->getStatus()==100 ||
	  QReport_JetPhi[jtyp]->getStatus()==200) 
	qr_Jet_Phi[jtyp] = 1;
      else if (QReport_JetPhi[jtyp]->getStatus()==300)
	qr_Jet_Phi[jtyp] = 0;
      else
	qr_Jet_Phi[jtyp] = -1;
    }
    else{ qr_Jet_Phi[jtyp] = -2;
    }
    //Jet Eta test
    if (QReport_JetEta[jtyp]){
      if (QReport_JetEta[jtyp]->getStatus()==100 ||
	  QReport_JetEta[jtyp]->getStatus()==200) 
	qr_Jet_Eta[jtyp] = 1;
      else if (QReport_JetEta[jtyp]->getStatus()==300) 
	qr_Jet_Eta[jtyp] = 0;
      else
	qr_Jet_Eta[jtyp] = -1;
    }
    else{ 
      qr_Jet_Eta[jtyp] = -2;
    }
      //Jet Constituents test
      if (QReport_JetConstituents[jtyp][0]){
      	if (QReport_JetConstituents[jtyp][0]->getStatus()==100 ||
	    QReport_JetConstituents[jtyp][0]->getStatus()==200) 
      	  qr_Jet_Constituents[jtyp][0] = 1;
	else if (QReport_JetConstituents[jtyp][0]->getStatus()==300) 
	  qr_Jet_Constituents[jtyp][0] = 0;
	else
	  qr_Jet_Constituents[jtyp][0] = -1;
      }
      else{ qr_Jet_Constituents[jtyp][0] = -2;
      }

      if (QReport_JetConstituents[jtyp][1]){
      	if (QReport_JetConstituents[jtyp][1]->getStatus()==100 ||
	    QReport_JetConstituents[jtyp][1]->getStatus()==200) 
      	  qr_Jet_Constituents[jtyp][1] = 1;
	else if (QReport_JetConstituents[jtyp][1]->getStatus()==300) 
	  qr_Jet_Constituents[jtyp][1] = 0;
	else
	  qr_Jet_Constituents[jtyp][1] = -1;
      }
      else{ qr_Jet_Constituents[jtyp][1] = -2;
      }
      //Jet EMFrac test
      if (QReport_JetEFrac[jtyp][0]){
	if (QReport_JetEFrac[jtyp][0]->getStatus()==100 ||
	    QReport_JetEFrac[jtyp][0]->getStatus()==200) 
	  qr_Jet_EFrac[jtyp][0] = 1;
	else if (QReport_JetEFrac[jtyp][0]->getStatus()==300) 
	  qr_Jet_EFrac[jtyp][0] = 0;
	else
	  qr_Jet_EFrac[jtyp][0] = -1;
      }
      else{ qr_Jet_EFrac[jtyp][0] = -2;
      }
      
      if (QReport_JetEFrac[jtyp][1]){
	if (QReport_JetEFrac[jtyp][1]->getStatus()==100 ||
	    QReport_JetEFrac[jtyp][1]->getStatus()==200) 
	  qr_Jet_EFrac[jtyp][1] = 1;
	else if (QReport_JetEFrac[jtyp][1]->getStatus()==300) 
	  qr_Jet_EFrac[jtyp][1] = 0;
	else
	  qr_Jet_EFrac[jtyp][1] = -1;
      }
      else{ qr_Jet_EFrac[jtyp][1] = -2;
      }
    
    if (verbose_) {
      printf("====================Jet Type %d QTest Report Summary========================\n",jtyp);
      printf("Eta:    Phi:   Pt 1:    2:    Const/Ntracks 1:    2:    EFrac/tracknhits 1:    2:\n");

      printf("%2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f    %2.2f\n", \
	     qr_Jet_Eta[jtyp],						\
	     qr_Jet_Phi[jtyp],						\
	     qr_Jet_Pt[jtyp][0],					\
	     qr_Jet_Pt[jtyp][1],					\
	     qr_Jet_Constituents[jtyp][0],				\
	     qr_Jet_Constituents[jtyp][1],				\
	     qr_Jet_EFrac[jtyp][0],					\
	     qr_Jet_EFrac[jtyp][1]);
      
    }
    //certification result for Jet

    //Only apply certain tests, as defined in the config
    for (int ttyp = 0; ttyp < 2;  ++ttyp) {
      if (!jetTests[jtyp][ttyp]) {
	qr_Jet_Pt[jtyp][ttyp]           = 1;
	if (ttyp ==1) {
	  qr_Jet_Eta[jtyp]          = 1;
	  qr_Jet_Phi[jtyp]          = 1;
	}
	qr_Jet_EFrac[jtyp][ttyp]        = 1;
	qr_Jet_Constituents[jtyp][ttyp] = 1;
      }
    }
    
    
  
    if ( (qr_Jet_EFrac[jtyp][0]        == 0) ||
	 (qr_Jet_EFrac[jtyp][1]        == 0) ||
	 (qr_Jet_Constituents[jtyp][1] == 0) || 
	 (qr_Jet_Constituents[jtyp][0] == 0) ||
	 (qr_Jet_Eta[jtyp]             == 0) ||
	 (qr_Jet_Phi[jtyp]             == 0) ||
	 (qr_Jet_Pt[jtyp][0]           == 0) ||
	 (qr_Jet_Pt[jtyp][1]           == 0)
	 )
      dc_Jet[jtyp] = 0;
    else if ( (qr_Jet_EFrac[jtyp][0]        == -1) &&
	      (qr_Jet_EFrac[jtyp][1]        == -1) &&
	      (qr_Jet_Constituents[jtyp][1] == -1) && 
	      (qr_Jet_Constituents[jtyp][0] == -1) &&
	      (qr_Jet_Eta[jtyp]             == -1) &&
	      (qr_Jet_Phi[jtyp]             == -1) &&
	      (qr_Jet_Pt[jtyp][0]           == -1) &&
	      (qr_Jet_Pt[jtyp][1]           == -1 )
	      )
      dc_Jet[jtyp] = -1;
    else if ( (qr_Jet_EFrac[jtyp][0]   == -2) &&
	      (qr_Jet_EFrac[jtyp][1]        == -2) &&
	      (qr_Jet_Constituents[jtyp][1] == -2) && 
	      (qr_Jet_Constituents[jtyp][0] == -2) &&
	      (qr_Jet_Eta[jtyp]             == -2) &&
	      (qr_Jet_Phi[jtyp]             == -2) &&
	      (qr_Jet_Pt[jtyp][0]           == -2) &&
	      (qr_Jet_Pt[jtyp][1]           == -2)
	      )
      dc_Jet[jtyp] = -2;
    else
      dc_Jet[jtyp] = 1;
    
    if (verbose_) std::cout<<"Certifying Jet algo: "<<jtyp<<" with value: "<<dc_Jet[jtyp]<<std::endl;

  
    CertificationSummaryMap->Fill(2, 4-jtyp, dc_Jet[jtyp]);
    reportSummaryMap->Fill(2, 4-jtyp, dc_Jet[jtyp]);
  }

  //-----------------------------
  // MET DQM Data Certification
  //-----------------------------
  //
  // Prepare test histograms
  //
  MonitorElement *meMExy[2][2];
  MonitorElement *meMEt[2];
  MonitorElement *meSumEt[2];
  MonitorElement *meMETPhi[2];
 
  RunDir = "";
  if (RunDir == "") newHistoName = "JetMET/MET/";
  else              newHistoName = RunDir+"/JetMET/Runsummary/MET/";

    metFolder = "Cleaned";
  
  //MEx/MEy monitor elements
  meMExy[0][0] = iget_.get(newHistoName+"met/"+metFolder+"/MEx");
  meMExy[0][1] = iget_.get(newHistoName+"met/"+metFolder+"/MEy");
  meMExy[1][0] = iget_.get(newHistoName+"pfMet/"+metFolder+"/MEx");
  meMExy[1][1] = iget_.get(newHistoName+"pfMet/"+metFolder+"/MEy");
 
  //MET Phi monitor elements
  meMETPhi[0]  = iget_.get(newHistoName+"met/"+metFolder+"/METPhi");
  meMETPhi[1]  = iget_.get(newHistoName+"pfMet/"+metFolder+"/METPhi");
  //MET monitor elements
  meMEt[0]  = iget_.get(newHistoName+"met/"+metFolder+"/MET");
  meMEt[1]  = iget_.get(newHistoName+"pfMet/"+metFolder+"/MET");
  //SumET monitor elements
  meSumEt[0]  = iget_.get(newHistoName+"met/"+metFolder+"/SumET");
  meSumEt[1]  = iget_.get(newHistoName+"pfMet/"+metFolder+"/SumET");
				   
  //----------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for MET
  //----------------------------------------------------------------------------

  // 2 types of MET {CaloMET, PfMET}  // It is 5 if CaloMETNoHF is included, 4 for MuonCorMET
  // removed 3rd type of TcMET
  // 2 types of tests Mean test/Kolmogorov test
  const QReport * QReport_MExy[2][2][2]={{{0}}};
  const QReport * QReport_MEt[2][2]={{0}};
  const QReport * QReport_SumEt[2][2]={{0}};
  //2 types of tests phiQTest and Kolmogorov test
  const QReport * QReport_METPhi[2][2]={{0}};


  float qr_MET_MExy[2][2][2] = {{{-999.}}};
  float qr_MET_MEt[2][2]     = {{-999.}};
  float qr_MET_SumEt[2][2]   = {{-999.}};
  float qr_MET_METPhi[2][2]  = {{-999.}};
  float dc_MET[2]            = {-999.};


  // J.Piedra, 27/02/212
  // removed MuCorrMET & TcMET --> loop up to 2 instead of 4, remove already from definition
  for (int mtyp = 0; mtyp < 2; ++mtyp){
    //Mean test results
    if (meMExy[mtyp][0] && meMExy[mtyp][0]->getRootObject()) {
      QReport_MExy[mtyp][0][0] = meMExy[mtyp][0]->getQReport("meanMExyTest");
      QReport_MExy[mtyp][1][0] = meMExy[mtyp][0]->getQReport("KolmogorovTest");
    }
    if (meMExy[mtyp][1]&& meMExy[mtyp][1]->getRootObject()) {
      QReport_MExy[mtyp][0][1] = meMExy[mtyp][1]->getQReport("meanMExyTest");
      QReport_MExy[mtyp][1][1] = meMExy[mtyp][1]->getQReport("KolmogorovTest");
    }
    if (meMEt[mtyp] && meMEt[mtyp]->getRootObject()) {
      QReport_MEt[mtyp][0]     = meMEt[mtyp]->getQReport("meanMETTest");
      QReport_MEt[mtyp][1]     = meMEt[mtyp]->getQReport("KolmogorovTest");
    }

    if (meSumEt[mtyp] && meSumEt[mtyp]->getRootObject()) {
      QReport_SumEt[mtyp][0]   = meSumEt[mtyp]->getQReport("meanSumETTest");
      QReport_SumEt[mtyp][1]   = meSumEt[mtyp]->getQReport("KolmogorovTest");
    }

    if (meMETPhi[mtyp] && meMETPhi[mtyp]->getRootObject()) {
      QReport_METPhi[mtyp][0]  = meMETPhi[mtyp]->getQReport("phiQTest");
      QReport_METPhi[mtyp][1]  = meMETPhi[mtyp]->getQReport("KolmogorovTest");
    }    
    for (int testtyp = 0; testtyp < 2; ++testtyp) {
      //MEx test
      if (QReport_MExy[mtyp][testtyp][0]){
	if (QReport_MExy[mtyp][testtyp][0]->getStatus()==100 ||
	    QReport_MExy[mtyp][testtyp][0]->getStatus()==200) 
	  qr_MET_MExy[mtyp][testtyp][0] = 1;
	else if (QReport_MExy[mtyp][testtyp][0]->getStatus()==300) 
	  qr_MET_MExy[mtyp][testtyp][0] = 0;
	else
	  qr_MET_MExy[mtyp][testtyp][0] = -1;
      }
      else qr_MET_MExy[mtyp][testtyp][0] = -2;
      //MEy test
      if (QReport_MExy[mtyp][testtyp][1]){
	if (QReport_MExy[mtyp][testtyp][1]->getStatus()==100 ||
	    QReport_MExy[mtyp][testtyp][1]->getStatus()==200) 
	  qr_MET_MExy[mtyp][testtyp][1] = 1;
	else if (QReport_MExy[mtyp][testtyp][1]->getStatus()==300) 
	  qr_MET_MExy[mtyp][testtyp][1] = 0;
	else
	  qr_MET_MExy[mtyp][testtyp][1] = -1;
      }
      else qr_MET_MExy[mtyp][testtyp][1] = -2;
      
      //MEt test
      if (QReport_MEt[mtyp][testtyp]){
	if (QReport_MEt[mtyp][testtyp]->getStatus()==100 ||
	    QReport_MEt[mtyp][testtyp]->getStatus()==200) 
	  qr_MET_MEt[mtyp][testtyp] = 1;
	else if (QReport_MEt[mtyp][testtyp]->getStatus()==300) 
	  qr_MET_MEt[mtyp][testtyp] = 0;
	else
	  qr_MET_MEt[mtyp][testtyp] = -1;
      }
      else{
	qr_MET_MEt[mtyp][testtyp] = -2;
      }
      //SumEt test
      if (QReport_SumEt[mtyp][testtyp]){
	if (QReport_SumEt[mtyp][testtyp]->getStatus()==100 ||
	    QReport_SumEt[mtyp][testtyp]->getStatus()==200) 
	  qr_MET_SumEt[mtyp][testtyp] = 1;
	else if (QReport_SumEt[mtyp][testtyp]->getStatus()==300) 
	  qr_MET_SumEt[mtyp][testtyp] = 0;
	else
	  qr_MET_SumEt[mtyp][testtyp] = -1;
      }
      else{
	qr_MET_SumEt[mtyp][testtyp] = -2;
      }
      //METPhi test
      if (QReport_METPhi[mtyp][testtyp]){
	if (QReport_METPhi[mtyp][testtyp]->getStatus()==100 ||
	    QReport_METPhi[mtyp][testtyp]->getStatus()==200) 
	  qr_MET_METPhi[mtyp][testtyp] = 1;
	else if (QReport_METPhi[mtyp][testtyp]->getStatus()==300) 
	  qr_MET_METPhi[mtyp][testtyp] = 0;
	else
	  qr_MET_METPhi[mtyp][testtyp] = -1;
      }
      else{
	qr_MET_METPhi[mtyp][testtyp] = -2;
      }
    }
 

    if (verbose_) {
      //certification result for MET
      printf("====================MET Type %d QTest Report Summary========================\n",mtyp);
      printf("MEx test    MEy test    MEt test:    SumEt test:    METPhi test:\n");
      for (int tt = 0; tt < 2; ++tt) {
	printf("%2.2f    %2.2f    %2.2f    %2.2f    %2.2f\n",qr_MET_MExy[mtyp][tt][0], \
	       qr_MET_MExy[mtyp][tt][1],				\
	       qr_MET_MEt[mtyp][tt],					\
	       qr_MET_SumEt[mtyp][tt],					\
	       qr_MET_METPhi[mtyp][tt]);
      }
      printf("===========================================================================\n");
    }


    //Only apply certain tests, as defined in the config
    for (int ttyp = 0; ttyp < 2;  ++ttyp) {
      if (!metTests[mtyp][ttyp]) {
	qr_MET_MExy[mtyp][ttyp][0]   = 1;
	qr_MET_MExy[mtyp][ttyp][1]   = 1;
	qr_MET_MEt[mtyp][ttyp]       = 1;
	qr_MET_SumEt[mtyp][ttyp]     = 1;
	qr_MET_METPhi[mtyp][ttyp]    = 1;
      }
    }
    

    if ( 
	(qr_MET_MExy[mtyp][0][0] == 0) ||
	(qr_MET_MExy[mtyp][0][1] == 0) ||
	(qr_MET_MEt[mtyp][0]     == 0) ||
	(qr_MET_SumEt[mtyp][0]   == 0) ||
	(qr_MET_METPhi[mtyp][0]  == 0) ||
	(qr_MET_MExy[mtyp][1][0] == 0) ||
	(qr_MET_MExy[mtyp][1][1] == 0) ||
	(qr_MET_MEt[mtyp][1]     == 0) ||
	(qr_MET_SumEt[mtyp][1]   == 0) ||
	(qr_MET_METPhi[mtyp][1]  == 0)
	)
      dc_MET[mtyp] = 0;
    else if (
	     (qr_MET_MExy[mtyp][0][0] == -1) &&
	     (qr_MET_MExy[mtyp][0][1] == -1) &&
	     (qr_MET_MEt[mtyp][0]     == -1) &&
	     (qr_MET_SumEt[mtyp][0]   == -1) &&
	     (qr_MET_METPhi[mtyp][0]  == -1) &&
	     (qr_MET_MExy[mtyp][1][0] == -1) &&
	     (qr_MET_MExy[mtyp][1][1] == -1) &&
	     (qr_MET_MEt[mtyp][1]     == -1) &&
	     (qr_MET_SumEt[mtyp][1]   == -1) &&
	     (qr_MET_METPhi[mtyp][1]  == -1)
	     )
      dc_MET[mtyp] = -1;
    else if ( 
	(qr_MET_MExy[mtyp][0][0] == -2) &&
	(qr_MET_MExy[mtyp][0][1] == -2) &&
	(qr_MET_MEt[mtyp][0]     == -2) &&
	(qr_MET_SumEt[mtyp][0]   == -2) &&
	(qr_MET_METPhi[mtyp][0]  == -2) &&
	(qr_MET_MExy[mtyp][1][0] == -2) &&
	(qr_MET_MExy[mtyp][1][1] == -2) &&
	(qr_MET_MEt[mtyp][1]     == -2) &&
	(qr_MET_SumEt[mtyp][1]   == -2) &&
	(qr_MET_METPhi[mtyp][1]  == -2)
	)
      dc_MET[mtyp] = -2;
    else
      dc_MET[mtyp] = 1;

    if (verbose_) std::cout<<"Certifying MET algo: "<<mtyp<<" with value: "<<dc_MET[mtyp]<<std::endl;
    CertificationSummaryMap->Fill(1, 4-mtyp, dc_MET[mtyp]);
    reportSummaryMap->Fill(1, 4-mtyp, dc_MET[mtyp]);
  }

				   
  //----------------------------------------------------------------------------
  //--- Extract quality test results and fill data certification results for MET
  //----------------------------------------------------------------------------
  // Commenting out unused but initialized variables. [Suchandra Dutta]
  float dc_CT[3]     = {-2.};
  dc_CT[0]  = -2.;
  dc_CT[1]  = -2.;
  dc_CT[2]  = -2.;

  for (int cttyp = 0; cttyp < 3; ++cttyp) {
    
    if (verbose_) std::cout<<"Certifying CaloTowers with value: "<<dc_CT[cttyp]<<std::endl;
    CertificationSummaryMap->Fill(0, 4-cttyp, dc_CT[cttyp]);
    reportSummaryMap->Fill(0, 4-cttyp, dc_CT[cttyp]);
  }
  ibook_.setCurrentFolder("");  
}

//define this as a plug-in
//DEFINE_FWK_MODULE(DataCertificationJetMET);
