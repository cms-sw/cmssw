#include <iostream>
//

#include "DQMOffline/EGamma/interface/PhotonAnalyzer.h"


//#define TWOPI 6.283185308
// 

/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
 **  $Date: 2009/01/28 14:05:03 $ 
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **   Jamie Antonelli, U. of Notre Dame, US
 **     
 ***/



using namespace std;

 
PhotonAnalyzer::PhotonAnalyzer( const edm::ParameterSet& pset ) 
{
 
    fName_              = pset.getUntrackedParameter<std::string>("Name");
    verbosity_          = pset.getUntrackedParameter<int>("Verbosity");

    prescaleFactor_     = pset.getUntrackedParameter<int>("prescaleFactor",1);
    
    photonProducer_     = pset.getParameter<std::string>("phoProducer");
    photonCollection_   = pset.getParameter<std::string>("photonCollection");

    triggerResultsHLT_     = pset.getParameter<edm::InputTag>("triggerResultsHLT");
    triggerResultsFU_     = pset.getParameter<edm::InputTag>("triggerResultsFU");

    minPhoEtCut_        = pset.getParameter<double>("minPhoEtCut");   

    cutStep_            = pset.getParameter<double>("cutStep");
    numberOfSteps_      = pset.getParameter<int>("numberOfSteps");

    useBinning_         = pset.getParameter<bool>("useBinning");
    useTriggerFiltering_= pset.getParameter<bool>("useTriggerFiltering");
    standAlone_         = pset.getParameter<bool>("standAlone");

    isolationStrength_  = pset.getParameter<int>("isolationStrength");


    parameters_ = pset;
   

}



PhotonAnalyzer::~PhotonAnalyzer() {




}


void PhotonAnalyzer::beginJob( const edm::EventSetup& setup)
{
  
  hltConfig_.init("HLT");

  

  nEvt_=0;
  nEntry_=0;
  
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  


 if (dbe_) {
    if (verbosity_ > 0 ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
  if (dbe_) {
    if (verbosity_ > 0 ) dbe_->showDirStructure();
  }



  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int etBin = parameters_.getParameter<int>("etBin");

  double sumMin = parameters_.getParameter<double>("sumMin");
  double sumMax = parameters_.getParameter<double>("sumMax");
  int sumBin = parameters_.getParameter<int>("sumBin");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");
 
  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");
 
  double r9Min = parameters_.getParameter<double>("r9Min"); 
  double r9Max = parameters_.getParameter<double>("r9Max"); 
  int r9Bin = parameters_.getParameter<int>("r9Bin");

  double hOverEMin = parameters_.getParameter<double>("hOverEMin"); 
  double hOverEMax = parameters_.getParameter<double>("hOverEMax"); 
  int hOverEBin = parameters_.getParameter<int>("hOverEBin");

  double xyMin = parameters_.getParameter<double>("xyMin"); 
  double xyMax = parameters_.getParameter<double>("xyMax"); 
  int xyBin = parameters_.getParameter<int>("xyBin");

  double numberMin = parameters_.getParameter<double>("numberMin"); 
  double numberMax = parameters_.getParameter<double>("numberMax"); 
  int numberBin = parameters_.getParameter<int>("numberBin");

  double zMin = parameters_.getParameter<double>("zMin"); 
  double zMax = parameters_.getParameter<double>("zMax"); 
  int zBin = parameters_.getParameter<int>("zBin");

  double rMin = parameters_.getParameter<double>("rMin"); 
  double rMax = parameters_.getParameter<double>("rMax"); 
  int rBin = parameters_.getParameter<int>("rBin");

  double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin"); 
  double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax"); 
  int dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin");

  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin"); 
  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax"); 
  int dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");


  vector<string> parts;
  parts.push_back("AllEcal");
  parts.push_back("Barrel");
  parts.push_back("Endcaps");


  vector<string> types;
  types.push_back("All");
  types.push_back("GoodCandidate");
  types.push_back("Background");

  //booking all histograms

  if (dbe_) {  

    currentFolder_.str("");
    currentFolder_ << "Egamma/PhotonAnalyzer/Efficiencies";
    dbe_->setCurrentFolder(currentFolder_.str());
    
        
    //Efficiency histograms

    h_phoEta_Loose_ = dbe_->book1D("phoEtaLoose"," Loose Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEta_Tight_ = dbe_->book1D("phoEtaTight"," Tight Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEt_Loose_ = dbe_->book1D("phoEtLoose"," Loose Photon Transverse Energy ", etBin,etMin, etMax);
    h_phoEt_Tight_ = dbe_->book1D("phoEtTight"," Tight Photon Transverse Energy ", etBin,etMin, etMax);

    p_efficiencyVsEtaLoose_ = dbe_->book1D("EfficiencyVsEtaLoose","Fraction of Loosely Isolated Photons  vs. Eta;#eta;",etaBin,etaMin, etaMax);
    p_efficiencyVsEtLoose_ = dbe_->book1D("EfficiencyVsEtLoose","Fraction of Loosely Isolated Photons vs. Et;Et (GeV)",etBin,etMin, etMax);
    p_efficiencyVsEtaTight_ = dbe_->book1D("EfficiencyVsEtaTight","Fraction of Tightly Isolated Photons  vs. Eta;#eta",etaBin,etaMin, etaMax);
    p_efficiencyVsEtTight_ = dbe_->book1D("EfficiencyVsEtTight","Fraction of Tightly Isolated Photons vs. Et;Et (GeV)",etBin,etMin, etMax);
  
    //Conversion fraction histograms
    
    p_convFractionVsEta_ = dbe_->book1D("convFractionVsEta","Fraction of Converted Photons  vs. Eta;#eta",etaBin,etaMin, etaMax);
    p_convFractionVsEt_ = dbe_->book1D("convFractionVsEt","Fraction of Converted Photons vs. Et;Et (GeV)",etBin,etMin, etMax);
    
    //Triggers passed
    
    h_triggers_ = dbe_->book1D("Triggers","Triggers Passed",500,0,500);


    for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
      
      // Photon histograms

      for(uint type=0;type!=types.size();++type){ //looping over isolation type
	
	currentFolder_.str("");
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
	dbe_->setCurrentFolder(currentFolder_.str());

	for(uint part=0;part!=parts.size();++part){ //loop over different parts of the ecal

	  h_phoE_part_.push_back(dbe_->book1D("phoE"+parts[part],types[type]+" Photon Energy: "+parts[part]+";E (GeV)", eBin,eMin, eMax));
	  h_phoEt_part_.push_back(dbe_->book1D("phoEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part]+";Et (GeV)", etBin,etMin, etMax));
	  h_r9_part_.push_back(dbe_->book1D("r9"+parts[part],types[type]+" Photon r9: "+parts[part]+";R9",r9Bin,r9Min, r9Max));
	  h_hOverE_part_.push_back(dbe_->book1D("hOverE"+parts[part],types[type]+" Photon H/E: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_h1OverE_part_.push_back(dbe_->book1D("h1OverE"+parts[part],types[type]+" Photon H/E for Depth 1: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_h2OverE_part_.push_back(dbe_->book1D("h2OverE"+parts[part],types[type]+" Photon H/E for Depth 2: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_nPho_part_.push_back(dbe_->book1D("nPho"+parts[part],"Number of "+types[type]+" Photons per Event: "+parts[part]+";# #gamma", 5,-0.5,4.5));
	}

	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionAllEcal","Distribution of "+types[type]+" Photons in Eta/Phi: AllEcal;#phi;#eta",phiBin,phiMin,phiMax,etaBin,etaMin,etaMax));
	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionBarrel","Distribution of "+types[type]+" Photons in Eta/Phi: Barrel;#phi;#eta",360,phiMin,phiMax,170,-1.5,1.5));
	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionEndcapMinus","Distribution of "+types[type]+" Photons in X/Y: EndcapMinus;x (cm);y (cm)",xyBin,xyMin,xyMax,xyBin,xyMin,xyMax));
	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionEndcapPlus","Distribution of "+types[type]+" Photons in X/Y: EndcapPlus;x (cm);y (cm)",xyBin,xyMin,xyMax,xyBin,xyMin,xyMax));

	h_phoE_isol_.push_back(h_phoE_part_);
	h_phoE_part_.clear();
	h_phoEt_isol_.push_back(h_phoEt_part_);
	h_phoEt_part_.clear();
	h_r9_isol_.push_back(h_r9_part_);
	h_r9_part_.clear();
	h_hOverE_isol_.push_back(h_hOverE_part_);
	h_hOverE_part_.clear();
	h_h1OverE_isol_.push_back(h_h1OverE_part_);
	h_h1OverE_part_.clear();
	h_h2OverE_isol_.push_back(h_h2OverE_part_);
	h_h2OverE_part_.clear();
	h_nPho_isol_.push_back(h_nPho_part_);
	h_nPho_part_.clear();

	h_phoDistribution_isol_.push_back(h_phoDistribution_part_);
	h_phoDistribution_part_.clear();

	h_phoEta_isol_.push_back(dbe_->book1D("phoEta",types[type]+" Photon Eta;#eta ",etaBin,etaMin, etaMax)) ;
	h_phoPhi_isol_.push_back(dbe_->book1D("phoPhi",types[type]+" Photon Phi;#phi ",phiBin,phiMin,phiMax)) ;
	h_r9VsEt_isol_.push_back(dbe_->book2D("r9VsEt2D",types[type]+" Photon r9 vs. Transverse Energy;Et (GeV);R9",etBin,etMin,etMax,r9Bin,r9Min,r9Max));
	p_r9VsEt_isol_.push_back(dbe_->book1D("r9VsEt",types[type]+" Photon r9 vs. Transverse Energy;Et (GeV);R9",etBin,etMin,etMax));

	h_phoSigmaIetaIeta_isol_.push_back(dbe_->book1D("phoSigmaIetaIeta",types[type]+" Photon #sigmai#etai#eta;#sigmai#etai#eta ",100,0,0.001)) ;
	h_sigmaIetaIetaVsEta_isol_.push_back(dbe_->book2D("sigmaIetaIetaVsEta2D",types[type]+" Photon #sigmai#etai#eta vs. #eta;#eta;#sigmai#etai#eta",etaBin,etaMin,etaMax,100,0,0.001));
	p_sigmaIetaIetaVsEta_isol_.push_back(dbe_->book1D("sigmaIetaIetaVsEta",types[type]+" Photon #sigmai#etai#eta vs. #eta;#eta;#sigmai#etai#eta",etaBin,etaMin,etaMax));

	// Isolation Variable infos

	h_nTrackIsolSolidVsEta_isol_.push_back(dbe_->book2D("nIsoTracksSolidVsEta2D","Avg Number Of Tracks in the Solid Iso Cone vs. #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax));
	h_trackPtSumSolidVsEta_isol_.push_back(dbe_->book2D("isoPtSumSolidVsEta2D","Avg Tracks Pt Sum in the Solid Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
	h_nTrackIsolHollowVsEta_isol_.push_back(dbe_->book2D("nIsoTracksHollowVsEta2D","Avg Number Of Tracks in the Hollow Iso Cone vs. #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax));
	h_trackPtSumHollowVsEta_isol_.push_back(dbe_->book2D("isoPtSumHollowVsEta2D","Avg Tracks Pt Sum in the Hollow Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
	h_ecalSumVsEta_isol_.push_back(dbe_->book2D("ecalSumVsEta2D","Avg Ecal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
	h_hcalSumVsEta_isol_.push_back(dbe_->book2D("hcalSumVsEta2D","Avg Hcal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
	p_nTrackIsolSolidVsEta_isol_.push_back(dbe_->book1D("nIsoTracksSolidVsEta","Avg Number Of Tracks in the Solid Iso Cone vs. #eta;#eta;# tracks",etaBin,etaMin, etaMax));
	p_trackPtSumSolidVsEta_isol_.push_back(dbe_->book1D("isoPtSumSolidVsEta","Avg Tracks Pt Sum in the Solid Iso Cone vs. #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax));
	p_nTrackIsolHollowVsEta_isol_.push_back(dbe_->book1D("nIsoTracksHollowVsEta","Avg Number Of Tracks in the Hollow Iso Cone vs. #eta;#eta;# tracks",etaBin,etaMin, etaMax));
	p_trackPtSumHollowVsEta_isol_.push_back(dbe_->book1D("isoPtSumHollowVsEta","Avg Tracks Pt Sum in the Hollow Iso Cone vs. #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax));
	p_ecalSumVsEta_isol_.push_back(dbe_->book1D("ecalSumVsEta","Avg Ecal Sum in the Iso Cone vs. #eta;#eta;E (GeV)",etaBin,etaMin, etaMax));
	p_hcalSumVsEta_isol_.push_back(dbe_->book1D("hcalSumVsEta","Avg Hcal Sum in the Iso Cone vs. #eta;#eta;E (GeV)",etaBin,etaMin, etaMax));

	h_nTrackIsolSolid_isol_.push_back(dbe_->book1D("nIsoTracksSolid","Avg Number Of Tracks in the Solid Iso Cone;# tracks",numberBin,numberMin,numberMax));
	h_trackPtSumSolid_isol_.push_back(dbe_->book1D("isoPtSumSolid","Avg Tracks Pt Sum in the Solid Iso Cone;Pt (GeV)",sumBin,sumMin,sumMax));
	h_nTrackIsolHollow_isol_.push_back(dbe_->book1D("nIsoTracksHollow","Avg Number Of Tracks in the Hollow Iso Cone;# tracks",numberBin,numberMin,numberMax));
	h_trackPtSumHollow_isol_.push_back(dbe_->book1D("isoPtSumHollow","Avg Tracks Pt Sum in the Hollow Iso Cone;Pt (GeV)",sumBin,sumMin,sumMax));
	h_ecalSum_isol_.push_back(dbe_->book1D("ecalSum","Avg Ecal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax));
	h_hcalSum_isol_.push_back(dbe_->book1D("hcalSum","Avg Hcal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax));


      }

      h_phoE_.push_back(h_phoE_isol_);
      h_phoE_isol_.clear();
      h_phoEt_.push_back(h_phoEt_isol_);
      h_phoEt_isol_.clear();
      h_r9_.push_back(h_r9_isol_);
      h_r9_isol_.clear();
      h_hOverE_.push_back(h_hOverE_isol_);
      h_hOverE_isol_.clear();
      h_h1OverE_.push_back(h_h1OverE_isol_);
      h_h1OverE_isol_.clear();
      h_h2OverE_.push_back(h_h2OverE_isol_);
      h_h2OverE_isol_.clear();

      h_nPho_.push_back(h_nPho_isol_);
      h_nPho_isol_.clear();
      
      h_phoDistribution_.push_back(h_phoDistribution_isol_);
      h_phoDistribution_isol_.clear();
            
      h_phoEta_.push_back(h_phoEta_isol_);
      h_phoEta_isol_.clear();
      h_phoPhi_.push_back(h_phoPhi_isol_);
      h_phoPhi_isol_.clear();
      h_r9VsEt_.push_back(h_r9VsEt_isol_);
      h_r9VsEt_isol_.clear();
      p_r9VsEt_.push_back(p_r9VsEt_isol_);
      p_r9VsEt_isol_.clear();
 
      h_phoSigmaIetaIeta_.push_back(h_phoSigmaIetaIeta_isol_);
      h_phoSigmaIetaIeta_isol_.clear();
  
      h_sigmaIetaIetaVsEta_.push_back(h_sigmaIetaIetaVsEta_isol_);
      h_sigmaIetaIetaVsEta_isol_.clear();
      p_sigmaIetaIetaVsEta_.push_back(p_sigmaIetaIetaVsEta_isol_);
      p_sigmaIetaIetaVsEta_isol_.clear();
 
      h_nTrackIsolSolidVsEta_.push_back(h_nTrackIsolSolidVsEta_isol_);
      h_trackPtSumSolidVsEta_.push_back(h_trackPtSumSolidVsEta_isol_);
      h_nTrackIsolHollowVsEta_.push_back(h_nTrackIsolHollowVsEta_isol_);
      h_trackPtSumHollowVsEta_.push_back(h_trackPtSumHollowVsEta_isol_);
      h_ecalSumVsEta_.push_back(h_ecalSumVsEta_isol_);
      h_hcalSumVsEta_.push_back(h_hcalSumVsEta_isol_);
      p_nTrackIsolSolidVsEta_.push_back(p_nTrackIsolSolidVsEta_isol_);
      p_trackPtSumSolidVsEta_.push_back(p_trackPtSumSolidVsEta_isol_);
      p_nTrackIsolHollowVsEta_.push_back(p_nTrackIsolHollowVsEta_isol_);
      p_trackPtSumHollowVsEta_.push_back(p_trackPtSumHollowVsEta_isol_);
      p_ecalSumVsEta_.push_back(p_ecalSumVsEta_isol_);
      p_hcalSumVsEta_.push_back(p_hcalSumVsEta_isol_);
    
      h_nTrackIsolSolidVsEta_isol_.clear();
      h_trackPtSumSolidVsEta_isol_.clear();
      h_nTrackIsolHollowVsEta_isol_.clear();
      h_trackPtSumHollowVsEta_isol_.clear();
      h_ecalSumVsEta_isol_.clear();
      h_hcalSumVsEta_isol_.clear();
      p_nTrackIsolSolidVsEta_isol_.clear();
      p_trackPtSumSolidVsEta_isol_.clear();
      p_nTrackIsolHollowVsEta_isol_.clear();
      p_trackPtSumHollowVsEta_isol_.clear();
      p_ecalSumVsEta_isol_.clear();
      p_hcalSumVsEta_isol_.clear();

      h_nTrackIsolSolid_.push_back(h_nTrackIsolSolid_isol_);
      h_trackPtSumSolid_.push_back(h_trackPtSumSolid_isol_);
      h_nTrackIsolHollow_.push_back(h_nTrackIsolHollow_isol_);
      h_trackPtSumHollow_.push_back(h_trackPtSumHollow_isol_);
      h_ecalSum_.push_back(h_ecalSum_isol_);
      h_hcalSum_.push_back(h_hcalSum_isol_);
      h_nTrackIsolSolid_isol_.clear();
      h_trackPtSumSolid_isol_.clear();
      h_nTrackIsolHollow_isol_.clear();
      h_trackPtSumHollow_isol_.clear();
      h_ecalSum_isol_.clear();
      h_hcalSum_isol_.clear();

      //conversion plots



      for(uint type=0;type!=types.size();++type){ //looping over isolation type

	currentFolder_.str("");	
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/Conversions";
	dbe_->setCurrentFolder(currentFolder_.str());

	for(uint part=0;part!=parts.size();++part){ //loop over different parts of the ecal

	  h_phoConvE_part_.push_back(dbe_->book1D("phoConvE"+parts[part],types[type]+" Photon Energy: "+parts[part]+";E (GeV)", eBin,eMin, eMax));
	  h_phoConvEt_part_.push_back(dbe_->book1D("phoConvEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part]+";Et (GeV)", etBin,etMin, etMax));

	  h_phoConvR9_part_.push_back(dbe_->book1D("phoConvR9"+parts[part],types[type]+" Photon r9: "+parts[part]+";R9",r9Bin,r9Min, r9Max));

	  h_nConv_part_.push_back(dbe_->book1D("nConv"+parts[part],"Number Of Conversions per Event:  "+parts[part]+";# conversions" ,numberBin,numberMin,numberMax));
	  h_eOverPTracks_part_.push_back(dbe_->book1D("eOverPTracks"+parts[part],"E/P of Conversions: "+parts[part]+";E/P" ,100, 0., 5.));
	  
	  h_dPhiTracksAtVtx_part_.push_back(dbe_->book1D("dPhiTracksAtVtx"+parts[part], "  #delta#phi of Conversion Tracks at Vertex: "+parts[part]+";#delta#phi",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax));
	  h_dCotTracks_part_.push_back(dbe_->book1D("dCotTracks"+parts[part],"#delta cotg(#Theta) of Conversion Tracks: "+parts[part]+";#deltacot(#theta)",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax)); 

	  h_dPhiTracksAtEcal_part_.push_back(dbe_->book1D("dPhiTracksAtEcal"+parts[part], "  #delta#phi of Conversion Tracks at Ecal: "+parts[part]+";#delta#phi",dPhiTracksBin,0.,dPhiTracksMax)); 
	  h_dEtaTracksAtEcal_part_.push_back(dbe_->book1D("dEtaTracksAtEcal"+parts[part], "  #delta#eta of Conversion Tracks at Ecal: "+parts[part]+";#delta#eta",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax)); 



	}
	h_phoConvE_isol_.push_back(h_phoConvE_part_);
	h_phoConvE_part_.clear();
	h_phoConvEt_isol_.push_back(h_phoConvEt_part_);
	h_phoConvEt_part_.clear();
	h_phoConvR9_isol_.push_back(h_phoConvR9_part_);
	h_phoConvR9_part_.clear();

	h_nConv_isol_.push_back(h_nConv_part_);
	h_nConv_part_.clear();
	h_eOverPTracks_isol_.push_back(h_eOverPTracks_part_);
	h_eOverPTracks_part_.clear();
	h_dPhiTracksAtVtx_isol_.push_back(h_dPhiTracksAtVtx_part_);
	h_dPhiTracksAtVtx_part_.clear();
	h_dCotTracks_isol_.push_back(h_dCotTracks_part_);
	h_dCotTracks_part_.clear();
	h_dPhiTracksAtEcal_isol_.push_back(h_dPhiTracksAtEcal_part_);
	h_dPhiTracksAtEcal_part_.clear();
	h_dEtaTracksAtEcal_isol_.push_back(h_dEtaTracksAtEcal_part_);
	h_dEtaTracksAtEcal_part_.clear();




	h_phoConvEta_isol_.push_back(dbe_->book1D("phoConvEta",types[type]+" Converted Photon Eta;#eta ",etaBin,etaMin, etaMax)) ;
	h_phoConvPhi_isol_.push_back(dbe_->book1D("phoConvPhi",types[type]+" Converted Photon Phi;#phi ",phiBin,phiMin,phiMax)) ;
	h_convVtxRvsZ_isol_.push_back(dbe_->book2D("convVtxRvsZ",types[type]+" Photon Reco conversion vtx position;Z (cm);R (cm)",zBin,zMin,zMax,rBin,rMin,rMax));
	h_convVtxRvsZLowEta_isol_.push_back(dbe_->book2D("convVtxRvsZLowEta",types[type]+" Photon Reco conversion vtx position: #eta < 1;Z (cm);R (cm)",zBin,zMin,zMax,rBin,rMin,rMax));
	h_convVtxRvsZHighEta_isol_.push_back(dbe_->book2D("convVtxRvsZHighEta",types[type]+" Photon Reco conversion vtx position: #eta > 1;Z (cm);R (cm)",zBin,zMin,zMax,rBin,rMin,rMax));
	h_nHitsVsEta_isol_.push_back(dbe_->book2D("nHitsVsEta2D",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta;#eta;# hits",etaBin,etaMin, etaMax,etaBin,0, 16));
	p_nHitsVsEta_isol_.push_back(dbe_->book1D("nHitsVsEta",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta;#eta;# hits",etaBin,etaMin, etaMax));	
	h_tkChi2_isol_.push_back(dbe_->book1D("tkChi2",types[type]+" Photons: Tracks from conversions: #chi^{2} of all tracks;#chi^{2}", 100, 0., 20.0));  
      }

      h_phoConvE_.push_back(h_phoConvE_isol_);
      h_phoConvE_isol_.clear();
      h_phoConvEt_.push_back(h_phoConvEt_isol_);
      h_phoConvEt_isol_.clear();
      h_phoConvR9_.push_back(h_phoConvR9_isol_);
      h_phoConvR9_isol_.clear();

      h_nConv_.push_back(h_nConv_isol_);
      h_nConv_isol_.clear();
      h_eOverPTracks_.push_back(h_eOverPTracks_isol_);
      h_eOverPTracks_isol_.clear();
      h_dPhiTracksAtVtx_.push_back(h_dPhiTracksAtVtx_isol_);
      h_dPhiTracksAtVtx_isol_.clear();
      h_dCotTracks_.push_back(h_dCotTracks_isol_);
      h_dCotTracks_isol_.clear();
      h_dPhiTracksAtEcal_.push_back(h_dPhiTracksAtEcal_isol_);
      h_dPhiTracksAtEcal_isol_.clear();  
      h_dEtaTracksAtEcal_.push_back(h_dEtaTracksAtEcal_isol_);
      h_dEtaTracksAtEcal_isol_.clear();

    
      h_phoConvEta_.push_back(h_phoConvEta_isol_);
      h_phoConvEta_isol_.clear();
      h_phoConvPhi_.push_back(h_phoConvPhi_isol_);
      h_phoConvPhi_isol_.clear();
      h_convVtxRvsZ_.push_back(h_convVtxRvsZ_isol_);
      h_convVtxRvsZ_isol_.clear();
      h_convVtxRvsZLowEta_.push_back(h_convVtxRvsZLowEta_isol_);
      h_convVtxRvsZLowEta_isol_.clear();
      h_convVtxRvsZHighEta_.push_back(h_convVtxRvsZHighEta_isol_);
      h_convVtxRvsZHighEta_isol_.clear();
      h_tkChi2_.push_back(h_tkChi2_isol_);
      h_tkChi2_isol_.clear();
      h_nHitsVsEta_.push_back(h_nHitsVsEta_isol_);
      h_nHitsVsEta_isol_.clear(); 
      p_nHitsVsEta_.push_back(p_nHitsVsEta_isol_);
      p_nHitsVsEta_isol_.clear();

 
    }


  } 

}
 
 




void PhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
 
  using namespace edm;
 
  if (nEvt_% prescaleFactor_ ) return; 
  nEvt_++;  
  LogInfo("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
 

  // Get the trigger information
  edm::Handle<edm::TriggerResults> triggerResultsHandle;
  e.getByLabel(triggerResultsHLT_,triggerResultsHandle);
  if(!triggerResultsHandle.isValid()) {
    edm::LogInfo("PhotonProducer") << "Error! Can't get the product "<<triggerResultsHLT_.label() << endl;; 
    e.getByLabel(triggerResultsFU_,triggerResultsHandle); 
    if(!triggerResultsHandle.isValid()) {
       edm::LogInfo("PhotonProducer") << "Error! Can't get the product  "<<triggerResultsFU_.label()<< endl;; 
      return;
    }
  }
  const edm::TriggerResults *triggerResults = triggerResultsHandle.product();

  // Get the recontructed  photons
  Handle<reco::PhotonCollection> photonHandle; 
  e.getByLabel(photonProducer_, photonCollection_ , photonHandle);
  if ( !photonHandle.isValid()) return;
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
 
  // grab PhotonId objects
  Handle<edm::ValueMap<bool> > loosePhotonFlag;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDLoose", loosePhotonFlag);
  Handle<edm::ValueMap<bool> > tightPhotonFlag;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDTight", tightPhotonFlag);


  // Create array to hold #photons/event information
  int nPho[100][3][3];

  for (int cut=0; cut!=100; ++cut){
    for (int type=0; type!=3; ++type){
      for (int part=0; part!=3; ++part){
	nPho[cut][type][part] = 0;
      }
    }
  }

  int photonCounter = 0;
  const edm::ValueMap<bool> *loosePhotonID = loosePhotonFlag.product();
  const edm::ValueMap<bool> *tightPhotonID = tightPhotonFlag.product();



  //  seeing if a photon trigger path was accepted


  //  getting photon-related triggers from the event
  vector<string> triggerNames;
  for(uint i=0;i<hltConfig_.size();++i){
    string trigger = hltConfig_.triggerName(i);
    if( trigger.find ("Photon") != std::string::npos)
      triggerNames.push_back(trigger);
  }
  

  //setting triggers histo bin labels
    TH1 *triggers = h_triggers_->getTH1();
  if(nEvt_ == 1){
    for(uint i=0;i<triggerNames.size();++i){
      string trigger = triggerNames[i];
      triggers->GetXaxis()->SetBinLabel(i+1,trigger.c_str());
    }
    triggers->GetXaxis()->SetRangeUser(0,triggerNames.size()-1);
  }

  //cutting out non-photon triggered events
  int AcceptsSum = 0;
  for (uint i=0; i<triggerNames.size();++i){
    const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerNames[i])); 
    if (triggerIndex < hltConfig_.size() ){
      AcceptsSum += triggerResults->accept(triggerIndex);
    }
  }
  if (AcceptsSum == 0 && useTriggerFiltering_) return;
 

  //  fill trigger histogram with which paths are accepted
  for (uint i=0; i<triggerNames.size();++i){
    const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerNames[i]));
    if (triggerIndex < hltConfig_.size() ){
      if (triggerResults->accept(triggerIndex)) h_triggers_->Fill(i);
    }
  }



  // Loop over all photons in event
  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {

    if ((*iPho).et()  < minPhoEtCut_) continue;
    


    edm::Ref<reco::PhotonCollection> photonref(photonHandle, photonCounter);
    photonCounter++;
    bool  isLoosePhoton = (*loosePhotonID)[photonref];
    bool  isTightPhoton = (*tightPhotonID)[photonref];


    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;
    bool  phoIsInEndcapMinus=false;
    bool  phoIsInEndcapPlus=false;
    float etaPho = (*iPho).superCluster()->eta();
    if ( fabs(etaPho) <  1.479 )
      phoIsInBarrel=true;
    else {
      phoIsInEndcap=true;
      if ( etaPho < 0.0 )
	phoIsInEndcapMinus=true;
      else
	phoIsInEndcapPlus=true;
    }


    //filling histograms to make isolation efficiencies
    if (isLoosePhoton){
	h_phoEta_Loose_->Fill( (*iPho).eta() );
	h_phoEt_Loose_->Fill( (*iPho).et() );
    }
    if (isTightPhoton){
	h_phoEta_Tight_->Fill( (*iPho).eta() );
	h_phoEt_Tight_->Fill( (*iPho).et() );
    }


    /////  From 30X Photons are already pre-selected at reconstruction level with a looseEM isolation
    bool isIsolated=false;
    if ( isolationStrength_ == 0)  isIsolated = isLoosePhoton;
    if ( isolationStrength_ == 1)  isIsolated = isTightPhoton; 

    int type=0;
    if ( isIsolated ) type=1;
    if ( !isIsolated ) type=2;


    nEntry_++;


    for (int cut=0; cut !=numberOfSteps_; ++cut) {
      double Et =  (*iPho).et();
      bool passesCuts = false;

      if ( useBinning_ && Et > cut*cutStep_ && ( Et < (cut+1)*cutStep_  | cut == numberOfSteps_-1 ) ){
	passesCuts = true;
      }
      else if ( !useBinning_ && Et > cut*cutStep_ ){
	passesCuts = true;
      }

      if (passesCuts){

	//filling isolation variable histograms
	h_nTrackIsolSolidVsEta_[cut][0]->Fill( (*iPho).eta(),(*iPho).nTrkSolidConeDR04());
	h_trackPtSumSolidVsEta_[cut][0]->Fill((*iPho).eta(), (*iPho).isolationTrkSolidConeDR04());
	
	h_nTrackIsolHollowVsEta_[cut][0]->Fill( (*iPho).eta(),(*iPho).nTrkHollowConeDR04());
	h_trackPtSumHollowVsEta_[cut][0]->Fill((*iPho).eta(), (*iPho).isolationTrkHollowConeDR04());
	
	h_ecalSumVsEta_[cut][0]->Fill((*iPho).eta(), (*iPho).ecalRecHitSumConeDR04());
	h_hcalSumVsEta_[cut][0]->Fill((*iPho).eta(), (*iPho).hcalTowerSumConeDR04());

	h_nTrackIsolSolid_[cut][0]->Fill( (*iPho).nTrkSolidConeDR04());
	h_trackPtSumSolid_[cut][0]->Fill((*iPho).isolationTrkSolidConeDR04());
	
	h_nTrackIsolHollow_[cut][0]->Fill((*iPho).nTrkHollowConeDR04());
	h_trackPtSumHollow_[cut][0]->Fill((*iPho).isolationTrkHollowConeDR04());
	
	h_ecalSum_[cut][0]->Fill((*iPho).ecalRecHitSumConeDR04());
	h_hcalSum_[cut][0]->Fill((*iPho).hcalTowerSumConeDR04());


	//filling all photons histograms
	h_phoE_[cut][0][0]->Fill( (*iPho).energy() );
	h_phoEt_[cut][0][0]->Fill( (*iPho).et() );

	h_r9_[cut][0][0]->Fill( (*iPho).r9() );
	h_hOverE_[cut][0][0]->Fill( (*iPho).hadronicOverEm() );
	h_h1OverE_[cut][0][0]->Fill( (*iPho).hadronicDepth1OverEm() );
	h_h2OverE_[cut][0][0]->Fill( (*iPho).hadronicDepth2OverEm() );



	nPho[cut][0][0]++;
	h_nConv_[cut][0][0]->Fill(float( (*iPho).conversions().size() ));

	h_phoDistribution_[cut][0][0]->Fill( (*iPho).phi(),(*iPho).eta() );

	h_phoEta_[cut][0]->Fill( (*iPho).eta() );
	h_phoPhi_[cut][0]->Fill( (*iPho).phi() );      

	h_r9VsEt_[cut][0]->Fill( (*iPho).et(), (*iPho).r9() );

	h_phoSigmaIetaIeta_[cut][0]->Fill( (*iPho).sigmaIetaIeta() );

	h_sigmaIetaIetaVsEta_[cut][0]->Fill( (*iPho).eta(),(*iPho).sigmaIetaIeta() );

	// iso/noniso photons histograms
	h_nTrackIsolSolidVsEta_[cut][type]->Fill( (*iPho).eta(),(*iPho).nTrkSolidConeDR04());
	h_trackPtSumSolidVsEta_[cut][type]->Fill((*iPho).eta(), (*iPho).isolationTrkSolidConeDR04());
	
	h_nTrackIsolHollowVsEta_[cut][type]->Fill( (*iPho).eta(),(*iPho).nTrkHollowConeDR04());
	h_trackPtSumHollowVsEta_[cut][type]->Fill((*iPho).eta(), (*iPho).isolationTrkHollowConeDR04());
	
	h_ecalSumVsEta_[cut][type]->Fill((*iPho).eta(), (*iPho).ecalRecHitSumConeDR04());
	h_hcalSumVsEta_[cut][type]->Fill((*iPho).eta(), (*iPho).hcalTowerSumConeDR04());

	h_nTrackIsolSolid_[cut][type]->Fill((*iPho).nTrkSolidConeDR04());
	h_trackPtSumSolid_[cut][type]->Fill((*iPho).isolationTrkSolidConeDR04());
	
	h_nTrackIsolHollow_[cut][type]->Fill((*iPho).nTrkHollowConeDR04());
	h_trackPtSumHollow_[cut][type]->Fill((*iPho).isolationTrkHollowConeDR04());
	
	h_ecalSum_[cut][type]->Fill((*iPho).ecalRecHitSumConeDR04());
	h_hcalSum_[cut][type]->Fill((*iPho).hcalTowerSumConeDR04());


	h_phoE_[cut][type][0]->Fill( (*iPho).energy() );
	h_phoEt_[cut][type][0]->Fill( (*iPho).et() );

	h_r9_[cut][type][0]->Fill( (*iPho).r9() );
	h_hOverE_[cut][type][0]->Fill( (*iPho).hadronicOverEm() );
	h_h1OverE_[cut][type][0]->Fill( (*iPho).hadronicDepth1OverEm() );
	h_h2OverE_[cut][type][0]->Fill( (*iPho).hadronicDepth2OverEm() );

	nPho[cut][type][0]++;
	h_nConv_[cut][type][0]->Fill(float( (*iPho).conversions().size() ));

	h_phoDistribution_[cut][type][0]->Fill( (*iPho).phi(),(*iPho).eta() );

	h_phoEta_[cut][type]->Fill( (*iPho).eta() );
	h_phoPhi_[cut][type]->Fill( (*iPho).phi() );      

	h_r9VsEt_[cut][type]->Fill( (*iPho).et(), (*iPho).r9() );

	h_phoSigmaIetaIeta_[cut][type]->Fill( (*iPho).sigmaIetaIeta() );

	h_sigmaIetaIetaVsEta_[cut][type]->Fill( (*iPho).eta(), (*iPho).sigmaIetaIeta() );


	if((*iPho).hasConversionTracks()){
	  	h_phoConvE_[cut][0][0]->Fill( (*iPho).energy() );
	  	h_phoConvE_[cut][type][0]->Fill( (*iPho).energy() );
	  	h_phoConvEt_[cut][0][0]->Fill( (*iPho).et() );
	  	h_phoConvEt_[cut][type][0]->Fill( (*iPho).et() );
	  	h_phoConvR9_[cut][0][0]->Fill( (*iPho).r9() );
	  	h_phoConvR9_[cut][type][0]->Fill( (*iPho).r9() );
	}

	//filling both types of histograms for different ecal parts
	int part = 0;
	if ( phoIsInBarrel )
	  part = 1;
	if ( phoIsInEndcap )
	  part = 2;

	if((*iPho).hasConversionTracks()){
	  	h_phoConvE_[cut][0][part]->Fill( (*iPho).energy() );
	  	h_phoConvE_[cut][type][part]->Fill( (*iPho).energy() );
	  	h_phoConvEt_[cut][0][part]->Fill( (*iPho).et() );
	  	h_phoConvEt_[cut][type][part]->Fill( (*iPho).et() );
	  	h_phoConvR9_[cut][0][part]->Fill( (*iPho).r9() );
	  	h_phoConvR9_[cut][type][part]->Fill( (*iPho).r9() );
	}

	h_phoE_[cut][0][part]->Fill( (*iPho).energy() );
	h_phoEt_[cut][0][part]->Fill( (*iPho).et() );

	h_r9_[cut][0][part]->Fill( (*iPho).r9() );
	h_hOverE_[cut][0][part]->Fill( (*iPho).hadronicOverEm() );
	h_h1OverE_[cut][0][part]->Fill( (*iPho).hadronicDepth1OverEm() );
	h_h2OverE_[cut][0][part]->Fill( (*iPho).hadronicDepth2OverEm() );


	nPho[cut][0][part]++;
	h_nConv_[cut][0][part]->Fill(float( (*iPho).conversions().size() ));

	if ( phoIsInBarrel )  h_phoDistribution_[cut][0][1]->Fill( (*iPho).phi(),(*iPho).eta() );
	if ( phoIsInEndcapMinus )  h_phoDistribution_[cut][0][2]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	if ( phoIsInEndcapPlus )  h_phoDistribution_[cut][0][3]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );


	h_phoE_[cut][type][part]->Fill( (*iPho).energy() );
	h_phoEt_[cut][type][part]->Fill( (*iPho).et() );

	h_r9_[cut][type][part]->Fill( (*iPho).r9() );
	h_hOverE_[cut][type][part]->Fill( (*iPho).hadronicOverEm() );
	h_h1OverE_[cut][type][part]->Fill( (*iPho).hadronicDepth1OverEm() );
	h_h2OverE_[cut][type][part]->Fill( (*iPho).hadronicDepth2OverEm() );

	nPho[cut][type][part]++;
	h_nConv_[cut][type][part]->Fill(float( (*iPho).conversions().size() ));

       	if ( phoIsInBarrel )  h_phoDistribution_[cut][type][1]->Fill( (*iPho).phi(),(*iPho).eta() );
	if ( phoIsInEndcapMinus )  h_phoDistribution_[cut][type][2]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	if ( phoIsInEndcapPlus )  h_phoDistribution_[cut][type][3]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );

	//loop over conversions

	std::vector<reco::ConversionRef> conversions = (*iPho).conversions();
	for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

	  reco::ConversionRef aConv=conversions[iConv];

	  if ( conversions[iConv]->nTracks() <2 ) continue; 


	  h_phoConvEta_[cut][0]->Fill( conversions[iConv]->caloCluster()[0]->eta()  );
	  h_phoConvPhi_[cut][0]->Fill( conversions[iConv]->caloCluster()[0]->phi()  );  
	  h_phoConvEta_[cut][type]->Fill( conversions[iConv]->caloCluster()[0]->eta()  );
	  h_phoConvPhi_[cut][type]->Fill( conversions[iConv]->caloCluster()[0]->phi()  );  


	  if ( conversions[iConv]->conversionVertex().isValid() ) {

	    h_convVtxRvsZ_[cut][0] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
					       sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	    h_convVtxRvsZ_[cut][type] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
					       sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;

	    if(fabs(conversions[iConv]->caloCluster()[0]->eta()) < 1){
	      h_convVtxRvsZLowEta_[cut][0] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
					      sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	      h_convVtxRvsZLowEta_[cut][type] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
						 sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	    }
	    else{
	      h_convVtxRvsZHighEta_[cut][0] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
					      sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	      h_convVtxRvsZHighEta_[cut][type] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
						 sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	    }

	  }


	  std::vector<reco::TrackRef> tracks = conversions[iConv]->tracks();

	  for (unsigned int i=0; i<tracks.size(); i++) {
	    h_tkChi2_[cut][0] ->Fill (tracks[i]->normalizedChi2()) ; 
	    h_tkChi2_[cut][type] ->Fill (tracks[i]->normalizedChi2()) ; 
	    h_nHitsVsEta_[cut][0]->Fill(  conversions[iConv]->caloCluster()[0]->eta(),   float(tracks[i]->numberOfValidHits() ) );
	    h_nHitsVsEta_[cut][type]->Fill(  conversions[iConv]->caloCluster()[0]->eta(),   float(tracks[i]->numberOfValidHits() ) );
	  }

	  float  DPhiTracksAtVtx = -99;
	  float  dPhiTracksAtEcal= -99;
	  float  dEtaTracksAtEcal= -99;


	  float phiTk1= tracks[0]->innerMomentum().phi();
	  float phiTk2= tracks[1]->innerMomentum().phi();
	  DPhiTracksAtVtx = phiTk1-phiTk2;
	  DPhiTracksAtVtx = phiNormalization( DPhiTracksAtVtx );


	  if (aConv->bcMatchingWithTracks()[0].isNonnull() && aConv->bcMatchingWithTracks()[1].isNonnull() ) {
	    float recoPhi1 = aConv->ecalImpactPosition()[0].phi();
	    float recoPhi2 = aConv->ecalImpactPosition()[1].phi();
	    float recoEta1 = aConv->ecalImpactPosition()[0].eta();
	    float recoEta2 = aConv->ecalImpactPosition()[1].eta();

	    recoPhi1 = phiNormalization(recoPhi1);
	    recoPhi2 = phiNormalization(recoPhi2);

	    dPhiTracksAtEcal = recoPhi1 -recoPhi2;
	    dPhiTracksAtEcal = phiNormalization( dPhiTracksAtEcal );
	    dEtaTracksAtEcal = recoEta1 -recoEta2;

	  }

	  h_eOverPTracks_[cut][0][0] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_eOverPTracks_[cut][type][0] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_dCotTracks_[cut][0][0] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dCotTracks_[cut][type][0] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dPhiTracksAtVtx_[cut][0][0]->Fill( DPhiTracksAtVtx);
	  h_dPhiTracksAtVtx_[cut][type][0]->Fill( DPhiTracksAtVtx);
	  h_dPhiTracksAtEcal_[cut][0][0]->Fill( fabs(dPhiTracksAtEcal));
	  h_dPhiTracksAtEcal_[cut][type][0]->Fill( fabs(dPhiTracksAtEcal));
	  h_dEtaTracksAtEcal_[cut][0][0]->Fill( dEtaTracksAtEcal);
	  h_dEtaTracksAtEcal_[cut][type][0]->Fill( dEtaTracksAtEcal);
	  
	  //filling both types of histograms for different ecal parts
	  int part = 0;
	  if ( phoIsInBarrel ) part = 1;
 	  if ( phoIsInEndcap ) part = 2;



	  h_eOverPTracks_[cut][0][part] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_eOverPTracks_[cut][type][part] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_dCotTracks_[cut][0][part] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dCotTracks_[cut][type][part] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dPhiTracksAtVtx_[cut][0][part]->Fill( DPhiTracksAtVtx);
	  h_dPhiTracksAtVtx_[cut][type][part]->Fill( DPhiTracksAtVtx);
	  h_dPhiTracksAtEcal_[cut][0][part]->Fill( fabs(dPhiTracksAtEcal));
	  h_dPhiTracksAtEcal_[cut][type][part]->Fill( fabs(dPhiTracksAtEcal));
	  h_dEtaTracksAtEcal_[cut][0][part]->Fill( dEtaTracksAtEcal);
	  h_dEtaTracksAtEcal_[cut][type][part]->Fill( dEtaTracksAtEcal);

	}//end loop over conversions

      }
    }
    
  }/// End loop over Reco  particles
    
  //filling number of photons per event histograms
  for (int cut=0; cut !=numberOfSteps_; ++cut) {
    for(int type=0;type!=3;++type){
      for(int part=0;part!=3;++part){
	h_nPho_[cut][type][part]-> Fill (float(nPho[cut][type][part]));
      }
    }
  }



}




void PhotonAnalyzer::endJob()
{
  
  if(standAlone_){

    vector<string> types;
    types.push_back("All");
    types.push_back("GoodCandidate");
    types.push_back("Background");
    
    std::string AllPath = "Egamma/PhotonAnalyzer/AllPhotons/";
    std::string IsoPath = "Egamma/PhotonAnalyzer/GoodCandidatePhotons/";
    std::string NonisoPath = "Egamma/PhotonAnalyzer/BackgroundPhotons/";
    std::string EffPath = "Egamma/PhotonAnalyzer/Efficiencies/";
    
    currentFolder_.str("");
    currentFolder_ << "Et above 0 GeV/";
    
    dividePlots(dbe_->get(EffPath+"Triggers"),dbe_->get(EffPath+"Triggers"),nEvt_);
    
    //making efficiency plots
    
    dividePlots(dbe_->get(EffPath+"EfficiencyVsEtaLoose"),dbe_->get(EffPath+ "phoEtaLoose"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"));
    dividePlots(dbe_->get(EffPath+"EfficiencyVsEtLoose"),dbe_->get(EffPath+ "phoEtLoose"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"));
    dividePlots(dbe_->get(EffPath+"EfficiencyVsEtaTight"),dbe_->get(EffPath+ "phoEtaTight"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"));
    dividePlots(dbe_->get(EffPath+"EfficiencyVsEtTight"),dbe_->get(EffPath+ "phoEtTight"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"));
        
    //making conversion fraction plots
    
    dividePlots(dbe_->get(EffPath+"convFractionVsEta"),dbe_->get(AllPath+currentFolder_.str() +  "Conversions/phoConvEta"),dbe_->get(AllPath+currentFolder_.str() + "phoEta"));
    dividePlots(dbe_->get(EffPath+"convFractionVsEt"),dbe_->get(AllPath+currentFolder_.str() +  "Conversions/phoConvEtAllEcal"),dbe_->get(AllPath+currentFolder_.str() + "phoEtAllEcal"));
    
    currentFolder_.str("");
    currentFolder_ << EffPath;
    dbe_->setCurrentFolder(currentFolder_.str());

    dbe_->removeElement("phoEtaLoose");
    dbe_->removeElement("phoEtaTight");
    dbe_->removeElement("phoEtLoose");
    dbe_->removeElement("phoEtTight"); 
    


    for(uint type=0;type!=types.size();++type){
      
      for (int cut=0; cut !=numberOfSteps_; ++cut) {
	
	currentFolder_.str("");
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/";

	//making profiles
	
	doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksSolidVsEta2D"),dbe_->get(currentFolder_.str()+"nIsoTracksSolidVsEta"));
 	doProfileX( dbe_->get(currentFolder_.str()+"nIsoTracksHollowVsEta2D"), dbe_->get(currentFolder_.str()+"nIsoTracksHollowVsEta"));
	
	doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumSolidVsEta2D"), dbe_->get(currentFolder_.str()+"isoPtSumSolidVsEta"));
	doProfileX( dbe_->get(currentFolder_.str()+"isoPtSumHollowVsEta2D"), dbe_->get(currentFolder_.str()+"isoPtSumHollowVsEta"));
	
	doProfileX( dbe_->get(currentFolder_.str()+"ecalSumVsEta2D"), dbe_->get(currentFolder_.str()+"ecalSumVsEta"));
	doProfileX( dbe_->get(currentFolder_.str()+"hcalSumVsEta2D"), dbe_->get(currentFolder_.str()+"hcalSumVsEta"));

 	doProfileX( dbe_->get(currentFolder_.str()+"r9VsEt2D"),dbe_->get(currentFolder_.str()+"r9VsEt"));

 	doProfileX( dbe_->get(currentFolder_.str()+"sigmaIetaIetaVsEta2D"),dbe_->get(currentFolder_.str()+"sigmaIetaIetaVsEta"));
	
	//removing unneeded plots
	
	dbe_->setCurrentFolder(currentFolder_.str());
	
	dbe_->removeElement("nIsoTracksSolidVsEta2D");
 	dbe_->removeElement("nIsoTracksHollowVsEta2D");
	dbe_->removeElement("isoPtSumSolidVsEta2D");
	dbe_->removeElement("isoPtSumHollowVsEta2D");
	dbe_->removeElement("ecalSumVsEta2D");
	dbe_->removeElement("hcalSumVsEta2D");
 	dbe_->removeElement("r9VsEt2D");	
 	dbe_->removeElement("sigmaIetaIetaVsEta2D");
	
	//other plots
	
 	currentFolder_ << "Conversions/";
 	doProfileX( dbe_->get(currentFolder_.str()+"nHitsVsEta2D"),dbe_->get(currentFolder_.str()+"nHitsVsEta"));
 	dbe_->setCurrentFolder(currentFolder_.str());
 	dbe_->removeElement("nHitsVsEta2D");
	
      }
      
      
    }
    
  }


  bool outputMEsInRootFile = parameters_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
  
  edm::LogInfo("PhotonAnalyzer") << "Analyzed " << nEvt_  << "\n";
  return ;
}



float PhotonAnalyzer::phiNormalization(float & phi)
{
//---Definitions
 const float PI    = 3.1415927;
 const float TWOPI = 2.0*PI;


 if(phi >  PI) {phi = phi - TWOPI;}
 if(phi < -PI) {phi = phi + TWOPI;}

 return phi;

}




void PhotonAnalyzer::doProfileX(TH2 * th2, MonitorElement* me){

  if (th2->GetNbinsX()==me->getNbinsX()){
    TH1F * h1 = (TH1F*) th2->ProfileX();
    for (int bin=0;bin!=h1->GetNbinsX();bin++){
      me->setBinContent(bin+1,h1->GetBinContent(bin+1));
      me->setBinError(bin+1,h1->GetBinError(bin+1));
    }
    me->setEntries(h1->GetEntries());
    delete h1;
  } else {
    throw cms::Exception("PhotonAnalyzer") << "Different number of bins!";
  }
}

void PhotonAnalyzer::doProfileX(MonitorElement * th2m, MonitorElement* me) {

  doProfileX(th2m->getTH2F(), me);
}




void  PhotonAnalyzer::dividePlots(MonitorElement* dividend, MonitorElement* numerator, MonitorElement* denominator){
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
    dividend->setEntries(numerator->getEntries());
  }
}


void  PhotonAnalyzer::dividePlots(MonitorElement* dividend, MonitorElement* numerator, double denominator){
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
