#include <iostream>
//

#include "DQMOffline/EGamma/interface/PhotonOfflineClient.h"


//#define TWOPI 6.283185308
// 

/** \class PhotonOfflineClient
 **  
 **
 **  $Id: PhotonOfflineClient
 **  $Date: 2008/09/30 19:50:30 $ 
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **   Jamie Antonelli, U. of Notre Dame, US
 **     
 ***/



using namespace std;

 
PhotonOfflineClient::PhotonOfflineClient(const edm::ParameterSet& pset) 
{

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  parameters_ = pset;

  cutStep_            = pset.getParameter<double>("cutStep");
  numberOfSteps_      = pset.getParameter<int>("numberOfSteps");

}



PhotonOfflineClient::~PhotonOfflineClient()
{}

void PhotonOfflineClient::beginJob( const edm::EventSetup& setup)
{

}

void PhotonOfflineClient::analyze(const edm::Event& e, const edm::EventSetup& esup)
{}


void PhotonOfflineClient::endJob()
{}


void PhotonOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& setup)
{

  vector<string> types;
  types.push_back("All");
  types.push_back("Isolated");
  types.push_back("Nonisolated");

  std::string AllPath = "Egamma/PhotonAnalyzer/AllPhotons/";
  std::string IsoPath = "Egamma/PhotonAnalyzer/IsolatedPhotons/";
  std::string NonisoPath = "Egamma/PhotonAnalyzer/NonisolatedPhotons/";
  std::string IsoVarPath = "Egamma/PhotonAnalyzer/IsolationVariables/";

  dividePlots(dbe_->get("Egamma/PhotonAnalyzer/Triggers"),dbe_->get("Egamma/PhotonAnalyzer/Triggers"),dbe_->get(AllPath+"Et above 0 GeV/nPhoAllEcal")->getTH1F()->GetEntries());

  for (int cut=0; cut !=numberOfSteps_; ++cut) {

    currentFolder_.str("");
    currentFolder_ << "Et above " << cut*cutStep_ << " GeV/";

    //making efficiency plots
  
    dividePlots(dbe_->get(IsoVarPath+currentFolder_.str()+"EfficiencyVsEta"),dbe_->get(IsoPath+currentFolder_.str() + "phoEta"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"));
    dividePlots(dbe_->get(IsoVarPath+currentFolder_.str()+"EfficiencyVsEt"),dbe_->get(IsoPath+currentFolder_.str() + "phoEtAllEcal"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"));
 
    //making conversion fraction plots

    dividePlots(dbe_->get(IsoVarPath+currentFolder_.str()+"convFractionVsEta"),dbe_->get(AllPath+currentFolder_.str() + "Conversions/phoConvEta"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"));
    dividePlots(dbe_->get(IsoVarPath+currentFolder_.str()+"convFractionVsEt"),dbe_->get(AllPath+currentFolder_.str() + "Conversions/phoConvEtAllEcal"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"));
  


    //making isolation variable profiles
    currentFolder_.str("");
    currentFolder_ << IsoVarPath << "Et above " << cut*cutStep_ << " GeV/";
    dbe_->setCurrentFolder(currentFolder_.str());
  

 
 
    doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksSolid2D"),dbe_->get(currentFolder_.str()+"nIsoTracksSolid"));
    doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksHollow2D"), dbe_->get(currentFolder_.str()+"nIsoTracksHollow"));

    doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumSolid2D"), dbe_->get(currentFolder_.str()+"isoPtSumSolid"));
    doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumHollow2D"), dbe_->get(currentFolder_.str()+"isoPtSumHollow"));
  
    doProfileX( dbe_->get(currentFolder_.str()+"ecalSum2D"), dbe_->get(currentFolder_.str()+"ecalSum"));
    doProfileX( dbe_->get(currentFolder_.str()+"hcalSum2D"), dbe_->get(currentFolder_.str()+"hcalSum"));

//     //removing unneeded plots
   

    dbe_->removeElement("nIsoTracksSolid2D");
    dbe_->removeElement("nIsoTracksHollow2D");
    dbe_->removeElement("isoPtSumSolid2D");
    dbe_->removeElement("isoPtSumHollow2D");
    dbe_->removeElement("ecalSum2D");
    dbe_->removeElement("hcalSum2D");



 

    for(uint type=0;type!=types.size();++type){
      currentFolder_.str("");
      currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
   
      dbe_->setCurrentFolder(currentFolder_.str());
      doProfileX( dbe_->get(currentFolder_.str()+"/r9VsEt2D"),dbe_->get(currentFolder_.str()+"/r9VsEt"));
      currentFolder_ << "/Conversions";
      doProfileX( dbe_->get(currentFolder_.str()+"/nHitsVsEta2D"),dbe_->get(currentFolder_.str()+"/nHitsVsEta"));
    
      dbe_->removeElement("r9VsEt2D");
      dbe_->setCurrentFolder(currentFolder_.str());
      dbe_->removeElement("nHitsVsEta2D");
    }
    
  
  
  }
   
}


void PhotonOfflineClient::doProfileX(TH2 * th2, MonitorElement* me){

  if (th2->GetNbinsX()==me->getNbinsX()){
    TH1F * h1 = (TH1F*) th2->ProfileX();
    for (int bin=0;bin!=h1->GetNbinsX();bin++){
      me->setBinContent(bin+1,h1->GetBinContent(bin+1));
      me->setBinError(bin+1,h1->GetBinError(bin+1));
    }
    delete h1;
  } else {
    throw cms::Exception("PhotonOfflineClient") << "Different number of bins!";
  }
}

void PhotonOfflineClient::doProfileX(MonitorElement * th2m, MonitorElement* me) {

  doProfileX(th2m->getTH2F(), me);
}




void  PhotonOfflineClient::dividePlots(MonitorElement* dividend, MonitorElement* numerator, MonitorElement* denominator){
  double value,err;
  for (int j=1; j<=numerator->getNbinsX(); j++){
    if (denominator->getBinContent(j)!=0){
      value = ((double) numerator->getBinContent(j))/((double) denominator->getBinContent(j));
      err = sqrt( value*(1-value) / ((double) denominator->getBinContent(j)) );
      dividend->setBinContent(j, value);
      dividend->setBinError(j,err);
    }
    else {
      dividend->setBinContent(j, 0);
    }
  }
}


void  PhotonOfflineClient::dividePlots(MonitorElement* dividend, MonitorElement* numerator, double denominator){
  double value,err;

  for (int j=1; j<=numerator->getNbinsX(); j++){
    if (denominator!=0){
      value = ((double) numerator->getBinContent(j))/denominator;
      err = sqrt( value*(1-value) / denominator);
      dividend->setBinContent(j, value);
      dividend->setBinError(j,err);
    }
    else {
      dividend->setBinContent(j, 0);
    }
  }

}

