#include <iostream>
//

#include "DQMOffline/EGamma/interface/PhotonOfflineClient.h"


//#define TWOPI 6.283185308
// 

/** \class PhotonOfflineClient
 **  
 **
 **  $Id: PhotonOfflineClient
 **  $Date: 2010/05/13 20:12:50 $ 
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
  etMin = pset.getParameter<double>("etMin");
  etMax = pset.getParameter<double>("etMax");
  etBin = pset.getParameter<int>("etBin");
  etaMin = pset.getParameter<double>("etaMin");
  etaMax = pset.getParameter<double>("etaMax");
  etaBin = pset.getParameter<int>("etaBin");
  phiMin = pset.getParameter<double>("phiMin");
  phiMax = pset.getParameter<double>("phiMax");
  phiBin = pset.getParameter<int>("phiBin");

  standAlone_ = pset.getParameter<bool>("standAlone");
  batch_ = pset.getParameter<bool>("batch");
  outputFileName_ = pset.getParameter<string>("OutputFileName");
  inputFileName_  = pset.getUntrackedParameter<std::string>("InputFileName");


}



PhotonOfflineClient::~PhotonOfflineClient()
{}

void PhotonOfflineClient::beginJob()
{}

void PhotonOfflineClient::analyze(const edm::Event& e, const edm::EventSetup& esup)
{}


void PhotonOfflineClient::endJob()
{
  if(standAlone_) runClient();
}


void PhotonOfflineClient::endLuminosityBlock( const edm::LuminosityBlock& , const edm::EventSetup& setup)
{}

void PhotonOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& setup)
{
  if(!standAlone_) runClient();
}


void PhotonOfflineClient::runClient()
{

  if(!dbe_) return;

  if(batch_)  dbe_->open(inputFileName_);

  if(!dbe_->dirExists("Egamma/PhotonAnalyzer")){
    std::cout << "egamma directory doesn't exist..." << std::endl;
    return;
  }

  //setting variable bin sizes for E, Et plots
  vector<float> etBinVector;
  double etRange = etMax-etMin;
  etBinVector.push_back(0);
  for(int i=1;i!=etBin;++i){
    if(i<etBin/3.) etBinVector.push_back(etBinVector.back()+float(etRange/(2*etBin)));
    else if(i>=etBin/3. && i<etBin*(2./3.)) etBinVector.push_back(etBinVector.back()+float(2*etRange/(2*etBin)));
    else if(i>=etBin*(2./3.)) etBinVector.push_back(etBinVector.back()+float(3*etRange/(2*etBin)));
  }






  vector<string> types;
  types.push_back("All");
  types.push_back("GoodCandidate");
  types.push_back("Background");

  std::string AllPath = "Egamma/PhotonAnalyzer/AllPhotons/";
  std::string IsoPath = "Egamma/PhotonAnalyzer/GoodCandidatePhotons/";
  std::string NonisoPath = "Egamma/PhotonAnalyzer/BackgroundPhotons/";
  std::string EffPath = "Egamma/PhotonAnalyzer/Efficiencies/";
  std::string InvMassPath = "Egamma/PhotonAnalyzer/InvMass/";  


  //booking efficiency histograms

  currentFolder_.str("");
  currentFolder_ << "Egamma/PhotonAnalyzer/Efficiencies";
  dbe_->setCurrentFolder(currentFolder_.str()); 

  p_efficiencyVsEtaLoose_ = dbe_->book1D("EfficiencyVsEtaLoose","Fraction of Loosely Isolated Photons  vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtLoose_ = dbe_->book1D("EfficiencyVsEtLoose","Fraction of Loosely Isolated Photons vs. Et;Et (GeV)",etBin,etMin,etMax);
  p_efficiencyVsEtaTight_ = dbe_->book1D("EfficiencyVsEtaTight","Fraction of Tightly Isolated Photons  vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtTight_ = dbe_->book1D("EfficiencyVsEtTight","Fraction of Tightly Isolated Photons vs. Et;Et (GeV)",etBin,etMin,etMax);
  p_efficiencyVsEtaHLT_ = dbe_->book1D("EfficiencyVsEtaHLT","Fraction of Photons passing HLT vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtHLT_ = dbe_->book1D("EfficiencyVsEtHLT","Fraction of Photons passing HLT vs. Et;Et (GeV)",etBin,etMin,etMax);

  p_convFractionVsEtaLoose_ = dbe_->book1D("ConvFractionVsEtaLoose","Fraction of Loosely Isolated Photons with two tracks vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_convFractionVsEtLoose_ = dbe_->book1D("ConvFractionVsEtLoose","Fraction of Loosely Isolated Photons with two tracks vs. Et;Et (GeV)",etBin,etMin,etMax);
  p_convFractionVsEtaTight_ = dbe_->book1D("ConvFractionVsEtaTight","Fraction of Tightly Isolated Photons  with two tracks vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_convFractionVsEtTight_ = dbe_->book1D("ConvFractionVsEtTight","Fraction of Tightly Isolated Photons with two tracks vs. Et;Et (GeV)",etBin,etMin,etMax);

  
  p_vertexReconstructionEfficiencyVsEta_ = dbe_->book1D("VertexReconstructionEfficiencyVsEta","Fraction of Converted Photons having a valid vertex vs. Eta;#eta",etaBin,etaMin, etaMax);

  
  //booking conversion histograms
  
  for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
    for(uint type=0;type!=types.size();++type){ //looping over isolation type
      
      currentFolder_.str("");	
      currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/Conversions";
      dbe_->setCurrentFolder(currentFolder_.str());

      p_convFractionVsPhi_isol_.push_back(dbe_->book1D("convFractionVsPhi","Fraction of Converted Photons  vs. Phi;#phi",phiBin,phiMin, phiMax));      
      p_convFractionVsEta_isol_.push_back(dbe_->book1D("convFractionVsEta","Fraction of Converted Photons  vs. Eta;#eta",etaBin,etaMin, etaMax));
      p_convFractionVsEt_isol_.push_back(dbe_->book1D("convFractionVsEt","Fraction of Converted Photons vs. Et;Et (GeV)",etBin,etMin,etMax));

    }

    p_convFractionVsEt_.push_back(p_convFractionVsEt_isol_);
    p_convFractionVsEt_isol_.clear();
    p_convFractionVsEta_.push_back(p_convFractionVsEta_isol_);
    p_convFractionVsEta_isol_.clear(); 
    p_convFractionVsPhi_.push_back(p_convFractionVsPhi_isol_);
    p_convFractionVsPhi_isol_.clear(); 
    
 
  }

  //booking bad channels histograms

  for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
    for(uint type=0;type!=types.size();++type){ //looping over isolation type
      
      currentFolder_.str("");	
      currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
      dbe_->setCurrentFolder(currentFolder_.str());
      
      p_badChannelsFractionVsEta_isol_.push_back(dbe_->book1D("badChannelsFractionVsEta","Fraction of Photons with at least one bad channel vs. Eta;#eta",etaBin,etaMin, etaMax));
      p_badChannelsFractionVsEt_isol_.push_back(dbe_->book1D("badChannelsFractionVsEt","Fraction of Converted Photons with at least one bad channel vs. Et;Et (GeV)",etBin,etMin,etMax));
      p_badChannelsFractionVsPhi_isol_.push_back(dbe_->book1D("badChannelsFractionVsPhi","Fraction of Photons with at least one bad channel vs. Phi;#phi",phiBin,phiMin, phiMax));

    }

    p_badChannelsFractionVsEt_.push_back(p_badChannelsFractionVsEt_isol_);
    p_badChannelsFractionVsEt_isol_.clear();
    p_badChannelsFractionVsEta_.push_back(p_badChannelsFractionVsEta_isol_);
    p_badChannelsFractionVsEta_isol_.clear(); 
    p_badChannelsFractionVsPhi_.push_back(p_badChannelsFractionVsPhi_isol_);
    p_badChannelsFractionVsPhi_isol_.clear();    
 
  }
 



  currentFolder_.str("");
  currentFolder_ << "Et above 0 GeV/";

  dividePlots(dbe_->get(EffPath+"Filters"),dbe_->get(EffPath+"Filters"),dbe_->get(EffPath+ "phoEtHLT")->getTH1F()->GetEntries());

  //making efficiency plots
 
  dividePlots(dbe_->get(EffPath+"EfficiencyVsEtaLoose"),dbe_->get(EffPath+ "phoEtaLoose"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"));
  dividePlots(dbe_->get(EffPath+"EfficiencyVsEtLoose"),dbe_->get(EffPath+ "phoEtLoose"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"));
  dividePlots(dbe_->get(EffPath+"EfficiencyVsEtaTight"),dbe_->get(EffPath+ "phoEtaTight"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"));
  dividePlots(dbe_->get(EffPath+"EfficiencyVsEtTight"),dbe_->get(EffPath+ "phoEtTight"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"));
  dividePlots(dbe_->get(EffPath+"EfficiencyVsEtaHLT"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"),dbe_->get(EffPath+ "phoEtaHLT"));
  dividePlots(dbe_->get(EffPath+"EfficiencyVsEtHLT"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"),dbe_->get(EffPath+ "phoEtHLT")); 

  dividePlots(dbe_->get(EffPath+"ConvFractionVsEtaLoose"), dbe_->get(EffPath+ "convEtaLoose"), dbe_->get(EffPath+ "phoEtaLoose"));
  dividePlots(dbe_->get(EffPath+"ConvFractionVsEtLoose"), dbe_->get(EffPath+ "convEtLoose"), dbe_->get(EffPath+ "phoEtLoose"));
  dividePlots(dbe_->get(EffPath+"ConvFractionVsEtaTight"), dbe_->get(EffPath+ "convEtaTight"), dbe_->get(EffPath+ "phoEtaTight"));
  dividePlots(dbe_->get(EffPath+"ConvFractionVsEtTight"), dbe_->get(EffPath+ "convEtTight"), dbe_->get(EffPath+ "phoEtTight"));


  if(dbe_->get(EffPath + "phoEtaVertex")->getTH1F()->GetEntries() != 0 )
    dividePlots(dbe_->get(EffPath+"VertexReconstructionEfficiencyVsEta"),dbe_->get(AllPath+currentFolder_.str() + "Conversions/phoConvEta"),dbe_->get(EffPath + "phoEtaVertex"));



  currentFolder_.str("");
  currentFolder_ << EffPath;
  dbe_->setCurrentFolder(currentFolder_.str());
  
  dbe_->removeElement("phoEtaLoose");
  dbe_->removeElement("phoEtaTight");
  dbe_->removeElement("phoEtaHLT");
  dbe_->removeElement("phoEtLoose");
  dbe_->removeElement("phoEtTight"); 
  dbe_->removeElement("phoEtHLT");
  dbe_->removeElement("phoEtaVertex");

  dbe_->removeElement("convEtaLoose");
  dbe_->removeElement("convEtaTight");
  dbe_->removeElement("convEtLoose");
  dbe_->removeElement("convEtTight"); 



  for(uint type=0;type!=types.size();++type){
    
    for (int cut=0; cut !=numberOfSteps_; ++cut) {
      
      currentFolder_.str("");
      currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/";
      
      //making conversion fraction plots

      dividePlots(dbe_->get(currentFolder_.str()+"Conversions/convFractionVsEta"),dbe_->get(currentFolder_.str() +  "Conversions/phoConvEta"),dbe_->get(currentFolder_.str() + "phoEta"));
      dividePlots(dbe_->get(currentFolder_.str()+"Conversions/convFractionVsPhi"),dbe_->get(currentFolder_.str() +  "Conversions/phoConvPhi"),dbe_->get(currentFolder_.str() + "phoPhi"));
      dividePlots(dbe_->get(currentFolder_.str()+"Conversions/convFractionVsEt"),dbe_->get(currentFolder_.str() +  "Conversions/phoConvEtAllEcal"),dbe_->get(currentFolder_.str() + "phoEtAllEcal"));

      dividePlots(dbe_->get(currentFolder_.str()+"badChannelsFractionVsEt"),dbe_->get(currentFolder_.str() +  "phoEtBadChannels"),dbe_->get(currentFolder_.str() +  "phoEtAllEcal"));
      dividePlots(dbe_->get(currentFolder_.str()+"badChannelsFractionVsEta"),dbe_->get(currentFolder_.str() +  "phoEtaBadChannels"),dbe_->get(currentFolder_.str() +  "phoEta"));
      dividePlots(dbe_->get(currentFolder_.str()+"badChannelsFractionVsPhi"),dbe_->get(currentFolder_.str() +  "phoPhiBadChannels"),dbe_->get(currentFolder_.str() +  "phoPhi"));



      //removing unneeded plots
      
      dbe_->setCurrentFolder(currentFolder_.str());

      dbe_->removeElement("phoEtBadChannels");
      dbe_->removeElement("phoEtaBadChannels");
      dbe_->removeElement("phoPhiBadChannels");



//       dbe_->removeElement("nIsoTracksSolidVsEta2D");
//       dbe_->removeElement("nIsoTracksHollowVsEta2D");
//       dbe_->removeElement("isoPtSumSolidVsEta2D");
//       dbe_->removeElement("isoPtSumHollowVsEta2D");
//       dbe_->removeElement("ecalSumVsEta2D");
//       dbe_->removeElement("hcalSumVsEta2D");
//       dbe_->removeElement("nIsoTracksSolidVsEt2D");
//       dbe_->removeElement("nIsoTracksHollowVsEt2D");
//       dbe_->removeElement("isoPtSumSolidVsEt2D");
//       dbe_->removeElement("isoPtSumHollowVsEt2D");
//       dbe_->removeElement("ecalSumVsEt2D");
//       dbe_->removeElement("hcalSumVsEt2D");
//       dbe_->removeElement("r9VsEt2D");	
//       dbe_->removeElement("r9VsEta2D");
//       dbe_->removeElement("e1x5VsEt2D");	
//       dbe_->removeElement("e1x5VsEta2D");
//       dbe_->removeElement("e2x5VsEt2D");	
//       dbe_->removeElement("e2x5VsEta2D");
//       dbe_->removeElement("r1x5VsEt2D");	
//       dbe_->removeElement("r1x5VsEta2D");
//       dbe_->removeElement("r2x5VsEt2D");	
//       dbe_->removeElement("r2x5VsEta2D");	
//       dbe_->removeElement("sigmaIetaIetaVsEta2D");	

      
      //other plots


    }
    
    
  }
  

  if(standAlone_) dbe_->save(outputFileName_);
  else if(batch_) dbe_->save(inputFileName_);
 

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
      dividend->setBinError(j,0);
    }
    dividend->setEntries(numerator->getEntries());
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
