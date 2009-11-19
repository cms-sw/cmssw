#include "DQM/Physics/interface/MetChecker.h"


MetChecker::MetChecker(const edm::ParameterSet& iConfig, std::string relativePath, std::string label)
{
  //now do what ever initialization is needed
  dqmStore_ = edm::Service<DQMStore>().operator->();
  relativePath_  =   relativePath;
  label_  =   label;
}

MetChecker::~MetChecker()
{
  delete dqmStore_;
}

void
MetChecker::analyze(const std::vector<reco::CaloMET>& mets)
{
  if(mets.size()>0)
  {
    hists_["CaloMETInmHF"]->Fill((mets)[0].CaloMETInmHF());
    hists_["CaloMETInpHF"]->Fill((mets)[0].CaloMETInpHF());
    hists_["CaloMETPhiInmHF"]->Fill((mets)[0].CaloMETPhiInmHF());
    hists_["CaloMETPhiInpHF"]->Fill((mets)[0].CaloMETPhiInpHF());
    
    std::string histoname;
    
    hists_["emEtFraction"]		->Fill((mets)[0].emEtFraction());
    hists_["emEtInEB"] 		->Fill((mets)[0].emEtInEB());
    hists_["emEtInEE"] 		->Fill((mets)[0].emEtInEE());
    hists_["emEtInHF"] 	   	->Fill((mets)[0].emEtInHF());
    hists_["etFractionHadronic"]	->Fill((mets)[0].etFractionHadronic());
    hists_["hadEtInHB"] 		->Fill((mets)[0].hadEtInHB());
    hists_["hadEtInHE"] 		->Fill((mets)[0].hadEtInHE());
    hists_["hadEtInHF"] 		->Fill((mets)[0].hadEtInHF());
    hists_["hadEtInHO"] 		->Fill((mets)[0].hadEtInHO());
    hists_["maxEtInEmTowers"]	->Fill((mets)[0].maxEtInEmTowers());
    hists_["maxEtInHadTowers"]	->Fill((mets)[0].maxEtInHadTowers());
    hists_["metSignificance"] 	->Fill((mets)[0].metSignificance());
    
  }
}

void 
MetChecker::begin(const edm::EventSetup&)
{
  dqmStore_->setCurrentFolder( relativePath_+"/CaloMETs_"+label_ );
   
  hists_["CaloMETInmHF"] 		  = dqmStore_->book1D("CaloMETInmHF","ME_{T} in the forward (-) hadronic calorimeter",800,0,400);
  hists_["CaloMETInmHF"]->setAxisTitle("MET in ForwHCAL(-)",1);
  hists_["CaloMETInpHF"]		  = dqmStore_->book1D("CaloMETInpHF","ME_{T} in the forward (+) hadronic calorimeter",800,0,400);
  hists_["CaloMETInpHF"]->setAxisTitle("MET in ForwHCAL(+)",1);
  hists_["CaloMETPhiInmHF"]		  = dqmStore_->book1D("CaloMETPhiInmHF","ME_{T} Phi in the forward (-) hadronic calorimeter",400,-4,4);
  hists_["CaloMETPhiInmHF"]->setAxisTitle("#phi(MET) in ForwHCAL(-)",1);
  hists_["CaloMETPhiInpHF"]		  = dqmStore_->book1D("CaloMETPhiInpHF","ME_{T} Phi in the forward (+) hadronic calorimeter",400,-4,4);
  hists_["CaloMETPhiInpHF"]->setAxisTitle("#phi(MET) in ForwHCAL(+)",1);
  hists_["emEtFraction"] 	  = dqmStore_->book1D("emEtFraction","electromagnetic transverse energy fraction",100,0,1);
  hists_["emEtFraction"]->setAxisTitle("em energy fraction",1);
  hists_["emEtInEB"] 		  = dqmStore_->book1D("emEtInEB","electromagnetic transverse energy in the ECAL barrel",800,0,400);
  hists_["emEtInEB"]->setAxisTitle("em Et in ECAL barrel",1);
  hists_["emEtInEE"] 		  = dqmStore_->book1D("emEtInEE","electromagnetic transverse energy in the ECAL end-cap",800,0,400);
  hists_["emEtInEE"]->setAxisTitle("em Et in ECAL end-cap",1);
  hists_["emEtInHF"] 		  = dqmStore_->book1D("emEtInHF","electromagnetic transverse energy extracted from the forward HCAL",800,0,400);
  hists_["emEtInHF"]->setAxisTitle("em Et extracted in ForwHCAL",1);
  hists_["etFractionHadronic"]    = dqmStore_->book1D("etFractionHadronic","hadronic transverse energy fraction",100,0,1);
  hists_["etFractionHadronic"]->setAxisTitle("hcal Et fraction",1);
  hists_["hadEtInHB"] 		  = dqmStore_->book1D("hadEtInHB","hadronic transverse energy in the HCAL barrel",800,0,400);
  hists_["hadEtInHB"]->setAxisTitle("had Et in HCAL barrel",1);
  hists_["hadEtInHE"] 		  = dqmStore_->book1D("hadEtInHE","hadronic transverse energy in the HCAL end-cap",800,0,400);
  hists_["hadEtInHE"]->setAxisTitle("had Et in HCAL end-cap",1);
  hists_["hadEtInHF"] 		  = dqmStore_->book1D("hadEtInHF","hadronic transverse energy in the forward HCAL",800,0,400);
  hists_["hadEtInHF"]->setAxisTitle("had Et in HF",1);
  hists_["hadEtInHO"] 		  = dqmStore_->book1D("hadEtInHO","hadronic transverse energy in the forward HCAL",800,0,400);
  hists_["hadEtInHO"]->setAxisTitle("had Et in H0",1);
  hists_["maxEtInEmTowers"] 	  = dqmStore_->book1D("maxEtInEmTowers","Maximum energy deposited in ECAL towers",800,0,400);
  hists_["maxEtInEmTowers"]->setAxisTitle("max Et in EmTowers",1);
  hists_["maxEtInHadTowers"] 	  = dqmStore_->book1D("maxEtInHadTowers","Maximum energy deposited in HCAL towers",800,0,400);
  hists_["maxEtInHadTowers"]->setAxisTitle("max Et in HadTowers",1);
  hists_["metSignificance"] 	  = dqmStore_->book1D("metSignificance","Missing transverse energy significance ",200,-20,20);
  hists_["metSignificance"]->setAxisTitle("MET Significance",1);
}

void 
MetChecker::end() 
{
}

