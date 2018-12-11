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
  inputMETLabelRECOUncleaned_=iConfig.getParameter<edm::InputTag>("METTypeRECOUncleaned");
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

  isHI = conf_.getUntrackedParameter<bool>("isHI",false);
 
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
    if( ((subDirVecMET[i]+"/Uncleaned")==(RunDirMET+inputMETLabelRECOUncleaned_.label()+"/Uncleaned")) || ((subDirVecMET[i]+"/Uncleaned")==(RunDirMET+inputMETLabelMiniAOD_.label()+"/Uncleaned"))){
      //check filters in uncleaned directory
      std::string rundirMET_reco="";
      if((subDirVecMET[i]+"/Uncleaned")==(RunDirMET+inputMETLabelRECOUncleaned_.label()+"/Uncleaned")){
	rundirMET_reco = RunDirMET+inputMETLabelRECOUncleaned_.label()+"/Uncleaned";
      }else{
	rundirMET_reco = RunDirMET+inputMETLabelMiniAOD_.label()+"/Uncleaned";
      }
      MonitorElement* mMET_Reco=iget_.get(rundirMET_reco+"/"+"MET");
      MonitorElement* mMET_Reco_HBHENoiseFilter=iget_.get(rundirMET_reco+"/"+"MET_HBHENoiseFilter");
      MonitorElement* mMET_Reco_CSCTightHaloFilter=iget_.get(rundirMET_reco+"/"+"MET_CSCTightHaloFilter");
      MonitorElement* mMET_Reco_eeBadScFilter=iget_.get(rundirMET_reco+"/"+"MET_eeBadScFilter");
      MonitorElement* mMET_Reco_HBHEIsoNoiseFilter=iget_.get(rundirMET_reco+"/"+"MET_HBHEIsoNoiseFilter");
      MonitorElement* mMET_Reco_CSCTightHalo2015Filter=iget_.get(rundirMET_reco+"/"+"MET_CSCTightHalo2015Filter");
      MonitorElement* mMET_Reco_EcalDeadCellTriggerFilter=iget_.get(rundirMET_reco+"/"+"MET_EcalDeadCellTriggerFilter");
      MonitorElement* mMET_Reco_EcalDeadCellBoundaryFilter=iget_.get(rundirMET_reco+"/"+"MET_EcalDeadCellBoundaryFilter");
      MonitorElement* mMET_Reco_HcalStripHaloFilter=iget_.get(rundirMET_reco+"/"+"MET_HcalStripHaloFilter");
      ibook_.setCurrentFolder(rundirMET_reco);
      mMET_EffHBHENoiseFilter=ibook_.book1D("MET_EffHBHENoiseFilter",(TH1F*)mMET_Reco_HBHENoiseFilter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffHBHENoiseFilter->setBinContent(i,mMET_Reco_HBHENoiseFilter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffHBHENoiseFilter->setBinContent(i,0);
	}
      }
      mMET_EffCSCTightHaloFilter=ibook_.book1D("MET_EffCSCTightHaloFilter",(TH1F*)mMET_Reco_CSCTightHaloFilter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffCSCTightHaloFilter->setBinContent(i,mMET_Reco_CSCTightHaloFilter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffCSCTightHaloFilter->setBinContent(i,0);
	}
      }
      mMET_EffeeBadScFilter=ibook_.book1D("MET_EffeeBadScFilter",(TH1F*)mMET_Reco_eeBadScFilter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffeeBadScFilter->setBinContent(i,mMET_Reco_eeBadScFilter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffeeBadScFilter->setBinContent(i,0);
	}
      }
      mMET_EffHBHEIsoNoiseFilter=ibook_.book1D("MET_EffHBHEIsoNoiseFilter",(TH1F*)mMET_Reco_HBHEIsoNoiseFilter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffHBHEIsoNoiseFilter->setBinContent(i,mMET_Reco_HBHEIsoNoiseFilter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffHBHEIsoNoiseFilter->setBinContent(i,0);
	}
      }
      mMET_EffCSCTightHalo2015Filter=ibook_.book1D("MET_EffCSCTightHalo2015Filter",(TH1F*)mMET_Reco_CSCTightHalo2015Filter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffCSCTightHalo2015Filter->setBinContent(i,mMET_Reco_CSCTightHalo2015Filter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffCSCTightHalo2015Filter->setBinContent(i,0);
	}
      }
      mMET_EffEcalDeadCellTriggerFilter=ibook_.book1D("MET_EffEcalDeadCellTriggerFilter",(TH1F*)mMET_Reco_EcalDeadCellTriggerFilter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffEcalDeadCellTriggerFilter->setBinContent(i,mMET_Reco_EcalDeadCellTriggerFilter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffEcalDeadCellTriggerFilter->setBinContent(i,0);
	}
      }
      mMET_EffEcalDeadCellBoundaryFilter=ibook_.book1D("MET_EffEcalDeadCellBoundaryFilter",(TH1F*)mMET_Reco_EcalDeadCellBoundaryFilter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffEcalDeadCellBoundaryFilter->setBinContent(i,mMET_Reco_EcalDeadCellBoundaryFilter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffEcalDeadCellBoundaryFilter->setBinContent(i,0);
	}
      }
      mMET_EffHcalStripHaloFilter=ibook_.book1D("MET_EffHcalStripHaloFilter",(TH1F*)mMET_Reco_HcalStripHaloFilter->getRootObject());
      for(int i=0;i<=(mMET_Reco->getNbinsX()+1);i++){
	if(mMET_Reco->getBinContent(i)!=0){
	  mMET_EffHcalStripHaloFilter->setBinContent(i,mMET_Reco_HcalStripHaloFilter->getBinContent(i)/mMET_Reco->getBinContent(i));
	}else{
	  mMET_EffHcalStripHaloFilter->setBinContent(i,0);
	}
      }
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

    std::vector<MonitorElement*> me_MET_Reco;
    me_MET_Reco.push_back(mMET_Reco);
    me_MET_Reco.push_back(mMEy_Reco);
    me_MET_Reco.push_back(mSumET_Reco);
    me_MET_Reco.push_back(mMETPhi_Reco);
    me_MET_Reco.push_back(mMET_logx_Reco);
    me_MET_Reco.push_back(mSumET_logx_Reco);
    me_MET_Reco.push_back(mChargedHadronEtFraction_Reco);
    me_MET_Reco.push_back(mNeutralHadronEtFraction_Reco);
    me_MET_Reco.push_back(mPhotonEtFraction_Reco);
    me_MET_Reco.push_back(mHFHadronEtFraction_Reco);
    me_MET_Reco.push_back(mHFEMEtFraction_Reco);
    me_MET_Reco.push_back(mMET_nVtx_profile_Reco);
    me_MET_Reco.push_back(mSumET_nVtx_profile_Reco);
    me_MET_Reco.push_back(mChargedHadronEtFraction_nVtx_profile_Reco);
    me_MET_Reco.push_back(mNeutralHadronEtFraction_nVtx_profile_Reco);
    me_MET_Reco.push_back(mPhotonEtFraction_nVtx_profile_Reco);
    
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

    std::vector<MonitorElement*> me_MET_MiniAOD;
    me_MET_MiniAOD.push_back(mMET_MiniAOD);
    me_MET_MiniAOD.push_back(mMEy_MiniAOD);
    me_MET_MiniAOD.push_back(mSumET_MiniAOD);
    me_MET_MiniAOD.push_back(mMETPhi_MiniAOD);
    me_MET_MiniAOD.push_back(mMET_logx_MiniAOD);
    me_MET_MiniAOD.push_back(mSumET_logx_MiniAOD);
    me_MET_MiniAOD.push_back(mChargedHadronEtFraction_MiniAOD);
    me_MET_MiniAOD.push_back(mNeutralHadronEtFraction_MiniAOD);
    me_MET_MiniAOD.push_back(mPhotonEtFraction_MiniAOD);
    me_MET_MiniAOD.push_back(mHFHadronEtFraction_MiniAOD);
    me_MET_MiniAOD.push_back(mHFEMEtFraction_MiniAOD);
    me_MET_MiniAOD.push_back(mMET_nVtx_profile_MiniAOD);
    me_MET_MiniAOD.push_back(mSumET_nVtx_profile_MiniAOD);
    me_MET_MiniAOD.push_back(mChargedHadronEtFraction_nVtx_profile_MiniAOD);
    me_MET_MiniAOD.push_back(mNeutralHadronEtFraction_nVtx_profile_MiniAOD);
    me_MET_MiniAOD.push_back(mPhotonEtFraction_nVtx_profile_MiniAOD);

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
    mSumET_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("SumET_profile_MiniAOD_over_RECO","SumET_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);
    mChargedHadronEtFraction_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("PfChargedHadronEtFraction_profile_MiniAOD_over_RECO","PfChargedHadronEtFraction_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);
    mNeutralHadronEtFraction_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("PfNeutralHadronEtFraction_profile_MiniAOD_over_RECO","PfNeutralHadronEtFraction_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);
    mPhotonEtFraction_nVtx_profile_MiniAOD_over_Reco=ibook_.book1D("PfPhotonEtFraction_profile_MiniAOD_over_RECO","PfPhotonEtFraction_vs_nVtx",nbinsPV_, nPVMin_, nPVMax_);

    std::vector<MonitorElement*> me_MET_MiniAOD_over_Reco;
    me_MET_MiniAOD_over_Reco.push_back(mMET_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mMEy_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mSumET_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mMETPhi_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mMET_logx_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mSumET_logx_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mChargedHadronEtFraction_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mNeutralHadronEtFraction_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mPhotonEtFraction_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mHFHadronEtFraction_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mHFEMEtFraction_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mMET_nVtx_profile_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mSumET_nVtx_profile_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mChargedHadronEtFraction_nVtx_profile_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mNeutralHadronEtFraction_nVtx_profile_MiniAOD_over_Reco);
    me_MET_MiniAOD_over_Reco.push_back(mPhotonEtFraction_nVtx_profile_MiniAOD_over_Reco);

    for(unsigned int j=0;j<me_MET_MiniAOD_over_Reco.size();j++){
      MonitorElement* monMETReco=me_MET_Reco[j];if(monMETReco && monMETReco->getRootObject()){
	MonitorElement* monMETMiniAOD=me_MET_MiniAOD[j];if(monMETMiniAOD && monMETMiniAOD->getRootObject()){
	  MonitorElement* monMETMiniAOD_over_RECO=me_MET_MiniAOD_over_Reco[j];if(monMETMiniAOD_over_RECO && monMETMiniAOD_over_RECO->getRootObject()){
	    for(int i=0;i<=(monMETMiniAOD_over_RECO->getNbinsX()+1);i++){
	      if(monMETReco->getBinContent(i)!=0){
		monMETMiniAOD_over_RECO->setBinContent(i,monMETMiniAOD->getBinContent(i)/monMETReco->getBinContent(i));
	      }else if (monMETMiniAOD->getBinContent(i)!=0){
		monMETMiniAOD_over_RECO->setBinContent(i,-0.5);
	      }
	    }
	  }
	}
      }
    }
  }//check for RECO and MiniAOD directories

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

    std::vector<MonitorElement*> me_Jet_Reco;
    me_Jet_Reco.push_back(mPt_Reco);
    me_Jet_Reco.push_back(mEta_Reco);
    me_Jet_Reco.push_back(mPhi_Reco);
    me_Jet_Reco.push_back(mNjets_Reco);
    me_Jet_Reco.push_back(mPt_uncor_Reco);
    me_Jet_Reco.push_back(mEta_uncor_Reco);
    me_Jet_Reco.push_back(mPhi_uncor_Reco);
    me_Jet_Reco.push_back(mJetEnergyCorr_Reco);
    me_Jet_Reco.push_back(mJetEnergyCorrVSeta_Reco);
    me_Jet_Reco.push_back(mDPhi_Reco);
    me_Jet_Reco.push_back(mLooseJIDPassFractionVSeta_Reco);
    me_Jet_Reco.push_back(mPt_Barrel_Reco);
    me_Jet_Reco.push_back(mPt_EndCap_Reco);
    me_Jet_Reco.push_back(mPt_Forward_Reco);
    me_Jet_Reco.push_back(mMVAPUJIDDiscriminant_lowPt_Barrel_Reco);
    me_Jet_Reco.push_back(mMVAPUJIDDiscriminant_lowPt_EndCap_Reco);
    me_Jet_Reco.push_back(mMVAPUJIDDiscriminant_lowPt_Forward_Reco);
    me_Jet_Reco.push_back(mMVAPUJIDDiscriminant_mediumPt_EndCap_Reco);
    me_Jet_Reco.push_back(mMVAPUJIDDiscriminant_highPt_Barrel_Reco);
    me_Jet_Reco.push_back(mCHFracVSpT_Barrel_Reco);
    me_Jet_Reco.push_back(mNHFracVSpT_EndCap_Reco);
    me_Jet_Reco.push_back(mPhFracVSpT_Barrel_Reco);
    me_Jet_Reco.push_back(mHFHFracVSpT_Forward_Reco);
    me_Jet_Reco.push_back(mHFEFracVSpT_Forward_Reco);
    me_Jet_Reco.push_back(mCHFrac_Reco);
    me_Jet_Reco.push_back(mNHFrac_Reco);
    me_Jet_Reco.push_back(mPhFrac_Reco);
    me_Jet_Reco.push_back(mChargedMultiplicity_Reco);
    me_Jet_Reco.push_back(mNeutralMultiplicity_Reco);
    me_Jet_Reco.push_back(mMuonMultiplicity_Reco);
    me_Jet_Reco.push_back(mNeutralFraction_Reco);

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

    std::vector<MonitorElement*> me_Jet_MiniAOD;
    me_Jet_MiniAOD.push_back(mPt_MiniAOD);
    me_Jet_MiniAOD.push_back(mEta_MiniAOD);
    me_Jet_MiniAOD.push_back(mPhi_MiniAOD);
    me_Jet_MiniAOD.push_back(mNjets_MiniAOD);
    me_Jet_MiniAOD.push_back(mPt_uncor_MiniAOD);
    me_Jet_MiniAOD.push_back(mEta_uncor_MiniAOD);
    me_Jet_MiniAOD.push_back(mPhi_uncor_MiniAOD);
    me_Jet_MiniAOD.push_back(mJetEnergyCorr_MiniAOD);
    me_Jet_MiniAOD.push_back(mJetEnergyCorrVSeta_MiniAOD);
    me_Jet_MiniAOD.push_back(mDPhi_MiniAOD);
    me_Jet_MiniAOD.push_back(mLooseJIDPassFractionVSeta_MiniAOD);
    me_Jet_MiniAOD.push_back(mPt_Barrel_MiniAOD);
    me_Jet_MiniAOD.push_back(mPt_EndCap_MiniAOD);
    me_Jet_MiniAOD.push_back(mPt_Forward_MiniAOD);
    me_Jet_MiniAOD.push_back(mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD);
    me_Jet_MiniAOD.push_back(mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD);
    me_Jet_MiniAOD.push_back(mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD);
    me_Jet_MiniAOD.push_back(mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD);
    me_Jet_MiniAOD.push_back(mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD);
    me_Jet_MiniAOD.push_back(mCHFracVSpT_Barrel_MiniAOD);
    me_Jet_MiniAOD.push_back(mNHFracVSpT_EndCap_MiniAOD);
    me_Jet_MiniAOD.push_back(mPhFracVSpT_Barrel_MiniAOD);
    me_Jet_MiniAOD.push_back(mHFHFracVSpT_Forward_MiniAOD);
    me_Jet_MiniAOD.push_back(mHFEFracVSpT_Forward_MiniAOD);
    me_Jet_MiniAOD.push_back(mCHFrac_MiniAOD);
    me_Jet_MiniAOD.push_back(mNHFrac_MiniAOD);
    me_Jet_MiniAOD.push_back(mPhFrac_MiniAOD);
    me_Jet_MiniAOD.push_back(mChargedMultiplicity_MiniAOD);
    me_Jet_MiniAOD.push_back(mNeutralMultiplicity_MiniAOD);
    me_Jet_MiniAOD.push_back(mMuonMultiplicity_MiniAOD);
    me_Jet_MiniAOD.push_back(mNeutralFraction_MiniAOD);

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
    ibook_.setCurrentFolder(RunDirJet+"MiniAOD_over_RECO"+"/"+"DiJet");
    mCHFrac_MiniAOD_over_Reco=ibook_.book1D("CHFrac_MiniAOD_over_RECO",(TH1F*)mCHFrac_Reco->getRootObject());
    mNHFrac_MiniAOD_over_Reco=ibook_.book1D("NHFrac_MiniAOD_over_RECO",(TH1F*)mNHFrac_Reco->getRootObject());
    mPhFrac_MiniAOD_over_Reco=ibook_.book1D("PhFrac_MiniAOD_over_RECO",(TH1F*)mPhFrac_Reco->getRootObject());
    mChargedMultiplicity_MiniAOD_over_Reco=ibook_.book1D("ChargedMultiplicity_MiniAOD_over_RECO",(TH1F*)mChargedMultiplicity_Reco->getRootObject());
    mNeutralMultiplicity_MiniAOD_over_Reco=ibook_.book1D("NeutralMultiplicity_MiniAOD_over_RECO",(TH1F*)mNeutralMultiplicity_Reco->getRootObject());
    mMuonMultiplicity_MiniAOD_over_Reco=ibook_.book1D("MuonMultiplicity_MiniAOD_over_RECO",(TH1F*)mMuonMultiplicity_Reco->getRootObject());
    mNeutralFraction_MiniAOD_over_Reco=ibook_.book1D("NeutralConstituentsFraction_MiniAOD_over_RECO",(TH1F*)mNeutralFraction_Reco->getRootObject());

    std::vector<MonitorElement*> me_Jet_MiniAOD_over_Reco;
    me_Jet_MiniAOD_over_Reco.push_back(mPt_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mEta_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPhi_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mNjets_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPt_uncor_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mEta_uncor_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPhi_uncor_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mJetEnergyCorr_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mJetEnergyCorrVSeta_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mDPhi_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mLooseJIDPassFractionVSeta_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPt_Barrel_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPt_EndCap_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPt_Forward_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mMVAPUJIDDiscriminant_lowPt_Barrel_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mMVAPUJIDDiscriminant_lowPt_EndCap_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mMVAPUJIDDiscriminant_lowPt_Forward_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mMVAPUJIDDiscriminant_mediumPt_EndCap_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mMVAPUJIDDiscriminant_highPt_Barrel_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mCHFracVSpT_Barrel_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mNHFracVSpT_EndCap_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPhFracVSpT_Barrel_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mHFHFracVSpT_Forward_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mHFEFracVSpT_Forward_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mCHFrac_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mNHFrac_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mPhFrac_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mChargedMultiplicity_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mNeutralMultiplicity_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mMuonMultiplicity_MiniAOD_over_Reco);
    me_Jet_MiniAOD_over_Reco.push_back(mNeutralFraction_MiniAOD_over_Reco);

    for(unsigned int j=0;j<me_Jet_MiniAOD_over_Reco.size();j++){
      MonitorElement* monJetReco=me_Jet_Reco[j];if(monJetReco && monJetReco->getRootObject()){
	MonitorElement* monJetMiniAOD=me_Jet_MiniAOD[j];if(monJetMiniAOD && monJetMiniAOD->getRootObject()){
	  MonitorElement* monJetMiniAOD_over_RECO=me_Jet_MiniAOD_over_Reco[j];if(monJetMiniAOD_over_RECO && monJetMiniAOD_over_RECO->getRootObject()){
	    for(int i=0;i<=(monJetMiniAOD_over_RECO->getNbinsX()+1);i++){
	      if(monJetReco->getBinContent(i)!=0){
		monJetMiniAOD_over_RECO->setBinContent(i,monJetMiniAOD->getBinContent(i)/monJetReco->getBinContent(i));
	      }else if (monJetMiniAOD->getBinContent(i)!=0){
		monJetMiniAOD_over_RECO->setBinContent(i,-0.5);
	      }
	    }
	  }
	}
      }
    }
  }//check for RECO and MiniAOD directories

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
  if (RunDir.empty()) newHistoName = "JetMET/Jet/";
  else              newHistoName = RunDir+"/JetMET/Runsummary/Jet/";
  std::string cleaningdir = "";
    cleaningdir = "Cleaned";
 
  // Read different histograms for PbPb and pp collisions

  if(isHI){ // Histograms for heavy ions

    newHistoName = "JetMET/HIJetValidation/";
    cleaningdir = "";

    //Jet Phi histos
    meJetPhi[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Cs4PFJets/Phi");
    meJetPhi[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu3PFJets/Phi");
    meJetPhi[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4PFJets/Phi");
    meJetPhi[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4CaloJets/Phi");

    //Jet Eta histos
    meJetEta[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Cs4PFJets/Eta");
    meJetEta[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu3PFJets/Eta");
    meJetEta[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4PFJets/Eta");
    meJetEta[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4CaloJets/Eta");

    //Jet Pt histos
    meJetPt[0]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"Cs4PFJets/Pt");
    meJetPt[1]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu3PFJets/Pt");
    meJetPt[2]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4PFJets/Pt");
    meJetPt[3]  = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4CaloJets/Pt");

    //Jet Constituents histos
    meJetConstituents[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Cs4PFJets/Constituents");
    meJetConstituents[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu3PFJets/Constituents");
    meJetConstituents[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4PFJets/Constituents");
    meJetConstituents[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"Pu4CaloJets/Constituents");
    
    //There are no jet EMFrac histograms for HI. Dummy paths will pass the tests by default
    meJetEMFrac[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"dummy/dummy");
    meJetEMFrac[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"dummy/dummy");
    meJetEMFrac[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"dummy/dummy");
    meJetEMFrac[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"dummy/dummy");


  } else { // Histograms for protons

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

    //Jet Constituents histos
    meJetConstituents[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Barrel");
    meJetConstituents[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_EndCap");
    meJetConstituents[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/Constituents_Forward");
    meJetConstituents[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/Constituents");
    
    //Jet EMFrac histos
    meJetEMFrac[0] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Barrel");
    meJetEMFrac[1] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_EndCap");
    meJetEMFrac[2] = iget_.get(newHistoName+cleaningdir+jetAlgo+"PFJets/EFrac_Forward");
    meJetEMFrac[3] = iget_.get(newHistoName+cleaningdir+jetAlgo+"CaloJets/EFrac");

  }
				   
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
  const QReport* QReport_JetEta[4] = {nullptr};
  const QReport* QReport_JetPhi[4] = {nullptr};
  // Mean and KS tests for Calo and PF jets
  const QReport* QReport_JetConstituents[4][2] = {{nullptr}};
  const QReport* QReport_JetEFrac[4][2]        = {{nullptr}};
  const QReport* QReport_JetPt[4][2]           = {{nullptr}};

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

  // There is nothing on the first row for HI, so mark the unfilled
  if(isHI){
    CertificationSummaryMap->Fill(2, 0, -2);
    reportSummaryMap->Fill(2, 0, -2);
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
  if (RunDir.empty()) newHistoName = "JetMET/MET/";
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
  const QReport * QReport_MExy[2][2][2]={{{nullptr}}};
  const QReport * QReport_MEt[2][2]={{nullptr}};
  const QReport * QReport_SumEt[2][2]={{nullptr}};
  //2 types of tests phiQTest and Kolmogorov test
  const QReport * QReport_METPhi[2][2]={{nullptr}};


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

  // There is nothing on the first three rows for HI, so mark them unfilled
  if(isHI){
    for(int i = 0; i < 3; i++){
      CertificationSummaryMap->Fill(1, i, -2);
      reportSummaryMap->Fill(1, i, -2);
    }
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
