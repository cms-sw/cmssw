#include <iostream>
//

#include "DQMOffline/EGamma/interface/PhotonOfflineClient.h"


//#define TWOPI 6.283185308
// 

/** \class PhotonOfflineClient
 **  
 **
 **  $Id: PhotonOfflineClient
 **  $Date: 2009/06/09 12:28:22 $ 
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

  standAlone_ = pset.getParameter<bool>("standAlone");
  batch_ = pset.getParameter<bool>("batch");
  outputFileName_ = pset.getParameter<string>("OutputFileName");
  inputFileName_  = pset.getUntrackedParameter<std::string>("InputFileName");


}



PhotonOfflineClient::~PhotonOfflineClient()
{}

void PhotonOfflineClient::beginJob( const edm::EventSetup& setup)
{

}

void PhotonOfflineClient::analyze(const edm::Event& e, const edm::EventSetup& esup)
{}


void PhotonOfflineClient::endJob()
{

  if(batch_)  dbe_->open(inputFileName_);

  vector<string> types;
  types.push_back("All");
  types.push_back("GoodCandidate");
  types.push_back("Background");

  std::string AllPath = "Egamma/PhotonAnalyzer/AllPhotons/";
  std::string IsoPath = "Egamma/PhotonAnalyzer/GoodCandidatePhotons/";
  std::string NonisoPath = "Egamma/PhotonAnalyzer/BackgroundPhotons/";
  std::string EffPath = "Egamma/PhotonAnalyzer/Efficiencies/";
  

  //booking efficiency histograms

  currentFolder_.str("");
  currentFolder_ << "Egamma/PhotonAnalyzer/Efficiencies";
  dbe_->setCurrentFolder(currentFolder_.str()); 

  p_efficiencyVsEtaLoose_ = dbe_->book1D("EfficiencyVsEtaLoose","Fraction of Loosely Isolated Photons  vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtLoose_ = dbe_->book1D("EfficiencyVsEtLoose","Fraction of Loosely Isolated Photons vs. Et;Et (GeV)",etBin,etMin, etMax);
  p_efficiencyVsEtaTight_ = dbe_->book1D("EfficiencyVsEtaTight","Fraction of Tightly Isolated Photons  vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtTight_ = dbe_->book1D("EfficiencyVsEtTight","Fraction of Tightly Isolated Photons vs. Et;Et (GeV)",etBin,etMin, etMax);
  p_efficiencyVsEtaHLT_ = dbe_->book1D("EfficiencyVsEtaHLT","Fraction of Photons passing HLT vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtHLT_ = dbe_->book1D("EfficiencyVsEtHLT","Fraction of Photons passing HLT vs. Et;Et (GeV)",etBin,etMin, etMax);

  p_convFractionVsEtaLoose_ = dbe_->book1D("ConvFractionVsEtaLoose","Fraction of Loosely Isolated Photons with two tracks vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_convFractionVsEtLoose_ = dbe_->book1D("ConvFractionVsEtLoose","Fraction of Loosely Isolated Photons with two tracks vs. Et;Et (GeV)",etBin,etMin, etMax);
  p_convFractionVsEtaTight_ = dbe_->book1D("ConvFractionVsEtaTight","Fraction of Tightly Isolated Photons  with two tracks vs. Eta;#eta",etaBin,etaMin, etaMax);
  p_convFractionVsEtTight_ = dbe_->book1D("ConvFractionVsEtTight","Fraction of Tightly Isolated Photons with two tracks vs. Et;Et (GeV)",etBin,etMin, etMax);

  
  p_vertexReconstructionEfficiencyVsEta_ = dbe_->book1D("VertexReconstructionEfficiencyVsEta","Fraction of Converted Photons having a valid vertex vs. Eta;#eta",etaBin,etaMin, etaMax);

  
  //booking conversion histograms
  
  for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
    for(uint type=0;type!=types.size();++type){ //looping over isolation type
      
      currentFolder_.str("");	
      currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/Conversions";
      dbe_->setCurrentFolder(currentFolder_.str());
      
      p_convFractionVsEta_isol_.push_back(dbe_->book1D("convFractionVsEta","Fraction of Converted Photons  vs. Eta;#eta",etaBin,etaMin, etaMax));
      p_convFractionVsEt_isol_.push_back(dbe_->book1D("convFractionVsEt","Fraction of Converted Photons vs. Et;Et (GeV)",etBin,etMin, etMax));




      p_nHitsVsEta_isol_.push_back(dbe_->book1D("nHitsVsEta",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta;#eta;# hits",etaBin,etaMin, etaMax));
      p_tkChi2VsEta_isol_.push_back(dbe_->book1D("tkChi2VsEta",types[type]+" Photons: Tracks from conversions: #chi^{2} vs Eta;#eta;#chi^{2}",etaBin,etaMin, etaMax));
      p_dCotTracksVsEta_isol_.push_back(dbe_->book1D("dCotTracksVsEta",types[type]+" #delta cotg(#Theta) of Conversion Tracks  vs Eta;#eta;#delta cotg(#Theta)",etaBin,etaMin, etaMax));


    }

    p_convFractionVsEt_.push_back(p_convFractionVsEt_isol_);
    p_convFractionVsEt_isol_.clear();
    p_convFractionVsEta_.push_back(p_convFractionVsEta_isol_);
    p_convFractionVsEta_isol_.clear(); 
    
    
    p_nHitsVsEta_.push_back(p_nHitsVsEta_isol_);
    p_nHitsVsEta_isol_.clear();
    
    p_tkChi2VsEta_.push_back(p_tkChi2VsEta_isol_);
    p_tkChi2VsEta_isol_.clear();
    
    p_dCotTracksVsEta_.push_back(p_dCotTracksVsEta_isol_);
    p_dCotTracksVsEta_isol_.clear();
      
  }


  //booking profiles

  for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
    for(uint type=0;type!=types.size();++type){ //looping over isolation type
      
      currentFolder_.str("");	
      currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
      dbe_->setCurrentFolder(currentFolder_.str());
      
      
      p_r9VsEt_isol_.push_back(dbe_->book1D("r9VsEt",types[type]+" Photon r9 vs. Transverse Energy;Et (GeV);R9",etBin,etMin,etMax));
      p_r9VsEta_isol_.push_back(dbe_->book1D("r9VsEta",types[type]+" Photon r9 vs. #eta;#eta;R9",etaBin,etaMin,etaMax));
      
      p_e1x5VsEt_isol_.push_back(dbe_->book1D("e1x5VsEt",types[type]+" Photon e1x5 vs. Transverse Energy;Et (GeV);E1X5",etBin,etMin,etMax));
      p_e1x5VsEta_isol_.push_back(dbe_->book1D("e1x5VsEta",types[type]+" Photon e1x5 vs. #eta;#eta;E1X5",etaBin,etaMin,etaMax));
      
      p_e2x5VsEt_isol_.push_back(dbe_->book1D("e2x5VsEt",types[type]+" Photon e2x5 vs. Transverse Energy;Et (GeV);E2X5",etBin,etMin,etMax));
      p_e2x5VsEta_isol_.push_back(dbe_->book1D("e2x5VsEta",types[type]+" Photon e2x5 vs. #eta;#eta;E2X5",etaBin,etaMin,etaMax));
      
      p_r1x5VsEt_isol_.push_back(dbe_->book1D("r1x5VsEt",types[type]+" Photon r1x5 vs. Transverse Energy;Et (GeV);R1X5",etBin,etMin,etMax));
      p_r1x5VsEta_isol_.push_back(dbe_->book1D("r1x5VsEta",types[type]+" Photon r1x5 vs. #eta;#eta;R1X5",etaBin,etaMin,etaMax));
      
      p_r2x5VsEt_isol_.push_back(dbe_->book1D("r2x5VsEt",types[type]+" Photon r2x5 vs. Transverse Energy;Et (GeV);R2X5",etBin,etMin,etMax));
      p_r2x5VsEta_isol_.push_back(dbe_->book1D("r2x5VsEta",types[type]+" Photon r2x5 vs. #eta;#eta;R2X5",etaBin,etaMin,etaMax));
      
      
      p_sigmaIetaIetaVsEta_isol_.push_back(dbe_->book1D("sigmaIetaIetaVsEta",types[type]+" Photon #sigmai#etai#eta vs. #eta;#eta;#sigmai#etai#eta",etaBin,etaMin,etaMax));
      p_sigmaEtaEtaVsEta_isol_.push_back(dbe_->book1D("sigmaEtaEtaVsEta",types[type]+" Photon #sigma#eta#eta vs. #eta;#eta;#sigma#eta#eta",etaBin,etaMin,etaMax));
      
      
      p_nTrackIsolSolidVsEta_isol_.push_back(dbe_->book1D("nIsoTracksSolidVsEta","Avg Number Of Tracks in the Solid Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax));
      p_trackPtSumSolidVsEta_isol_.push_back(dbe_->book1D("isoPtSumSolidVsEta","Avg Tracks Pt Sum in the Solid Iso Cone vs.  #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax));
      p_nTrackIsolHollowVsEta_isol_.push_back(dbe_->book1D("nIsoTracksHollowVsEta","Avg Number Of Tracks in the Hollow Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax));
      p_trackPtSumHollowVsEta_isol_.push_back(dbe_->book1D("isoPtSumHollowVsEta","Avg Tracks Pt Sum in the Hollow Iso Cone vs.  #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax));
      p_ecalSumVsEta_isol_.push_back(dbe_->book1D("ecalSumVsEta","Avg Ecal Sum in the Iso Cone vs.  #eta;#eta;E (GeV)",etaBin,etaMin, etaMax));
      p_hcalSumVsEta_isol_.push_back(dbe_->book1D("hcalSumVsEta","Avg Hcal Sum in the Iso Cone vs.  #eta;#eta;E (GeV)",etaBin,etaMin, etaMax));
      
      
      p_nTrackIsolSolidVsEt_isol_.push_back(dbe_->book1D("nIsoTracksSolidVsEt","Avg Number Of Tracks in the Solid Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin, etMax));
      p_trackPtSumSolidVsEt_isol_.push_back(dbe_->book1D("isoPtSumSolidVsEt","Avg Tracks Pt Sum in the Solid Iso Cone vs.  E_{T};E_{T};Pt (GeV)",etBin,etMin, etMax));
      p_nTrackIsolHollowVsEt_isol_.push_back(dbe_->book1D("nIsoTracksHollowVsEt","Avg Number Of Tracks in the Hollow Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin, etMax));
      p_trackPtSumHollowVsEt_isol_.push_back(dbe_->book1D("isoPtSumHollowVsEt","Avg Tracks Pt Sum in the Hollow Iso Cone vs.  E_{T};E_{T};Pt (GeV)",etBin,etMin, etMax));
      p_ecalSumVsEt_isol_.push_back(dbe_->book1D("ecalSumVsEt","Avg Ecal Sum in the Iso Cone vs.  E_{T};E_{T};E (GeV)",etBin,etMin, etMax));
      p_hcalSumVsEt_isol_.push_back(dbe_->book1D("hcalSumVsEt","Avg Hcal Sum in the Iso Cone vs.  E_{T};E_{T};E (GeV)",etBin,etMin, etMax));
      
      

    }
    
    p_r9VsEt_.push_back(p_r9VsEt_isol_);
    p_r9VsEt_isol_.clear();
    
    p_r9VsEta_.push_back(p_r9VsEta_isol_);
    p_r9VsEta_isol_.clear();
    
    p_e1x5VsEt_.push_back(p_e1x5VsEt_isol_);
    p_e1x5VsEt_isol_.clear();
    
    p_e1x5VsEta_.push_back(p_e1x5VsEta_isol_);
    p_e1x5VsEta_isol_.clear();
    
    p_e2x5VsEt_.push_back(p_e2x5VsEt_isol_);
    p_e2x5VsEt_isol_.clear();
    
    p_e2x5VsEta_.push_back(p_e2x5VsEta_isol_);
    p_e2x5VsEta_isol_.clear();
    
    p_r1x5VsEt_.push_back(p_r1x5VsEt_isol_);
    p_r1x5VsEt_isol_.clear();
    
    p_r1x5VsEta_.push_back(p_r1x5VsEta_isol_);
    p_r1x5VsEta_isol_.clear();
    
    p_r2x5VsEt_.push_back(p_r2x5VsEt_isol_);
    p_r2x5VsEt_isol_.clear();
    
    p_r2x5VsEta_.push_back(p_r2x5VsEta_isol_);
    p_r2x5VsEta_isol_.clear();
    
    p_sigmaIetaIetaVsEta_.push_back(p_sigmaIetaIetaVsEta_isol_);
    p_sigmaIetaIetaVsEta_isol_.clear();
    
    p_sigmaEtaEtaVsEta_.push_back(p_sigmaEtaEtaVsEta_isol_);
    p_sigmaEtaEtaVsEta_isol_.clear();
    
    p_nTrackIsolSolidVsEta_.push_back(p_nTrackIsolSolidVsEta_isol_);
    p_trackPtSumSolidVsEta_.push_back(p_trackPtSumSolidVsEta_isol_);
    p_nTrackIsolHollowVsEta_.push_back(p_nTrackIsolHollowVsEta_isol_);
    p_trackPtSumHollowVsEta_.push_back(p_trackPtSumHollowVsEta_isol_);
    p_ecalSumVsEta_.push_back(p_ecalSumVsEta_isol_);
    p_hcalSumVsEta_.push_back(p_hcalSumVsEta_isol_);
    
    p_nTrackIsolSolidVsEt_.push_back(p_nTrackIsolSolidVsEt_isol_);
    p_trackPtSumSolidVsEt_.push_back(p_trackPtSumSolidVsEt_isol_);
    p_nTrackIsolHollowVsEt_.push_back(p_nTrackIsolHollowVsEt_isol_);
    p_trackPtSumHollowVsEt_.push_back(p_trackPtSumHollowVsEt_isol_);
    p_ecalSumVsEt_.push_back(p_ecalSumVsEt_isol_);
    p_hcalSumVsEt_.push_back(p_hcalSumVsEt_isol_);

    
    p_nTrackIsolSolidVsEt_isol_.clear();
    p_trackPtSumSolidVsEt_isol_.clear();
    p_nTrackIsolHollowVsEt_isol_.clear();
    p_trackPtSumHollowVsEt_isol_.clear();
    p_ecalSumVsEt_isol_.clear();
    p_hcalSumVsEt_isol_.clear();
    
    p_nTrackIsolSolidVsEta_isol_.clear();
    p_trackPtSumSolidVsEta_isol_.clear();
    p_nTrackIsolHollowVsEta_isol_.clear();
    p_trackPtSumHollowVsEta_isol_.clear();
    p_ecalSumVsEta_isol_.clear();
    p_hcalSumVsEta_isol_.clear();

    


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


  if(dbe_->get(AllPath + currentFolder_.str() + "Conversions/phoConvEta")->getTH1F()->GetEntries() != 0 )
    dividePlots(dbe_->get(EffPath+"VertexReconstructionEfficiencyVsEta"),dbe_->get(EffPath + "phoEtaVertex"),dbe_->get(AllPath+currentFolder_.str() + "Conversions/phoConvEta"));



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
      dividePlots(dbe_->get(currentFolder_.str()+"Conversions/convFractionVsEt"),dbe_->get(currentFolder_.str() +  "Conversions/phoConvEtAllEcal"),dbe_->get(currentFolder_.str() + "phoEtAllEcal"));

      //making profiles
      doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksSolidVsEta2D"),dbe_->get(currentFolder_.str()+"nIsoTracksSolidVsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksHollowVsEta2D"), dbe_->get(currentFolder_.str()+"nIsoTracksHollowVsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumSolidVsEta2D"), dbe_->get(currentFolder_.str()+"isoPtSumSolidVsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumHollowVsEta2D"), dbe_->get(currentFolder_.str()+"isoPtSumHollowVsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"ecalSumVsEta2D"), dbe_->get(currentFolder_.str()+"ecalSumVsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"hcalSumVsEta2D"), dbe_->get(currentFolder_.str()+"hcalSumVsEta"));

      doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksSolidVsEt2D"),dbe_->get(currentFolder_.str()+"nIsoTracksSolidVsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksHollowVsEt2D"), dbe_->get(currentFolder_.str()+"nIsoTracksHollowVsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumSolidVsEt2D"), dbe_->get(currentFolder_.str()+"isoPtSumSolidVsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumHollowVsEt2D"), dbe_->get(currentFolder_.str()+"isoPtSumHollowVsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"ecalSumVsEt2D"), dbe_->get(currentFolder_.str()+"ecalSumVsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"hcalSumVsEt2D"), dbe_->get(currentFolder_.str()+"hcalSumVsEt"));

      doProfileX( dbe_->get(currentFolder_.str()+"r9VsEt2D"),dbe_->get(currentFolder_.str()+"r9VsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"r9VsEta2D"),dbe_->get(currentFolder_.str()+"r9VsEta"));

      doProfileX( dbe_->get(currentFolder_.str()+"e1x5VsEt2D"),dbe_->get(currentFolder_.str()+"e1x5VsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"e1x5VsEta2D"),dbe_->get(currentFolder_.str()+"e1x5VsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"e2x5VsEt2D"),dbe_->get(currentFolder_.str()+"e2x5VsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"e2x5VsEta2D"),dbe_->get(currentFolder_.str()+"e2x5VsEta"));

      doProfileX( dbe_->get(currentFolder_.str()+"r1x5VsEt2D"),dbe_->get(currentFolder_.str()+"r1x5VsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"r1x5VsEta2D"),dbe_->get(currentFolder_.str()+"r1x5VsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"r2x5VsEt2D"),dbe_->get(currentFolder_.str()+"r2x5VsEt"));
      doProfileX( dbe_->get(currentFolder_.str()+"r2x5VsEta2D"),dbe_->get(currentFolder_.str()+"r2x5VsEta"));

      doProfileX( dbe_->get(currentFolder_.str()+"sigmaIetaIetaVsEta2D"),dbe_->get(currentFolder_.str()+"sigmaIetaIetaVsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"sigmaEtaEtaVsEta2D"),dbe_->get(currentFolder_.str()+"sigmaEtaEtaVsEta"));

      //removing unneeded plots
      
      dbe_->setCurrentFolder(currentFolder_.str());
      
      dbe_->removeElement("nIsoTracksSolidVsEta2D");
      dbe_->removeElement("nIsoTracksHollowVsEta2D");
      dbe_->removeElement("isoPtSumSolidVsEta2D");
      dbe_->removeElement("isoPtSumHollowVsEta2D");
      dbe_->removeElement("ecalSumVsEta2D");
      dbe_->removeElement("hcalSumVsEta2D");
      dbe_->removeElement("nIsoTracksSolidVsEt2D");
      dbe_->removeElement("nIsoTracksHollowVsEt2D");
      dbe_->removeElement("isoPtSumSolidVsEt2D");
      dbe_->removeElement("isoPtSumHollowVsEt2D");
      dbe_->removeElement("ecalSumVsEt2D");
      dbe_->removeElement("hcalSumVsEt2D");
      dbe_->removeElement("r9VsEt2D");	
      dbe_->removeElement("r9VsEta2D");
      dbe_->removeElement("e1x5VsEt2D");	
      dbe_->removeElement("e1x5VsEta2D");
      dbe_->removeElement("e2x5VsEt2D");	
      dbe_->removeElement("e2x5VsEta2D");
      dbe_->removeElement("r1x5VsEt2D");	
      dbe_->removeElement("r1x5VsEta2D");
      dbe_->removeElement("r2x5VsEt2D");	
      dbe_->removeElement("r2x5VsEta2D");	
      dbe_->removeElement("sigmaIetaIetaVsEta2D");	
      dbe_->removeElement("sigmaEtaEtaVsEta2D");
      
      //other plots

      currentFolder_ << "Conversions/";

      doProfileX( dbe_->get(currentFolder_.str()+"nHitsVsEta2D"),dbe_->get(currentFolder_.str()+"nHitsVsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"tkChi2VsEta2D"),dbe_->get(currentFolder_.str()+"tkChi2VsEta"));
      doProfileX( dbe_->get(currentFolder_.str()+"dCotTracksVsEta2D"),dbe_->get(currentFolder_.str()+"dCotTracksVsEta"));
      dbe_->setCurrentFolder(currentFolder_.str());
      dbe_->removeElement("nHitsVsEta2D");
      dbe_->removeElement("tkChi2VsEta2D");
      dbe_->removeElement("dCotTracksVsEta2D");

    }
    
    
  }
  

  if(standAlone_) dbe_->save(outputFileName_);
  else if(batch_) dbe_->save(inputFileName_);
}


void PhotonOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& setup)
{

 
}


void PhotonOfflineClient::doProfileX(TH2 * th2, MonitorElement* me){

  if (th2->GetNbinsX()==me->getNbinsX()){
    TH1F * h1 = (TH1F*) th2->ProfileX();
    for (int bin=0;bin!=h1->GetNbinsX();bin++){
      me->setBinContent(bin+1,h1->GetBinContent(bin+1));
      me->setBinError(bin+1,h1->GetBinError(bin+1));
    }
    me->setEntries(h1->GetEntries());
    delete h1;
  } else {
    throw cms::Exception("PhotonOfflineClient") << "Different number of bins!\n";
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

