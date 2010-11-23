#include <iostream>
#include <iomanip>
//

#include "DQMOffline/EGamma/interface/PhotonAnalyzer.h"



//#define TWOPI 6.283185308
// 

/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
 **  $Date: 2010/09/28 09:40:19 $ 
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

    barrelRecHitProducer_ = pset.getParameter<std::string>("barrelRecHitProducer");
    barrelRecHitCollection_ = pset.getParameter<std::string>("barrelRecHitCollection");

    endcapRecHitProducer_ = pset.getParameter<std::string>("endcapRecHitProducer");
    endcapRecHitCollection_ = pset.getParameter<std::string>("endcapRecHitCollection");



    triggerEvent_       = pset.getParameter<edm::InputTag>("triggerEvent");

    minPhoEtCut_        = pset.getParameter<double>("minPhoEtCut");   
    invMassEtCut_        = pset.getParameter<double>("invMassEtCut");  

    cutStep_            = pset.getParameter<double>("cutStep");
    numberOfSteps_      = pset.getParameter<int>("numberOfSteps");

    useBinning_         = pset.getParameter<bool>("useBinning");
    useTriggerFiltering_= pset.getParameter<bool>("useTriggerFiltering");

    standAlone_         = pset.getParameter<bool>("standAlone");
    outputFileName_ = pset.getParameter<string>("OutputFileName");


    isolationStrength_  = pset.getParameter<int>("isolationStrength");


    isHeavyIon_          = pset.getUntrackedParameter<bool>("isHeavyIon",false);

    parameters_ = pset;
   

}



PhotonAnalyzer::~PhotonAnalyzer() {}


void PhotonAnalyzer::beginJob()
{
  
  nEvt_=0;
  nEntry_=0;
  
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  


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

  double xMin = parameters_.getParameter<double>("xMin"); 
  double xMax = parameters_.getParameter<double>("xMax"); 
  int xBin = parameters_.getParameter<int>("xBin");

  double yMin = parameters_.getParameter<double>("yMin"); 
  double yMax = parameters_.getParameter<double>("yMax"); 
  int yBin = parameters_.getParameter<int>("yBin");

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

  double sigmaIetaMin = parameters_.getParameter<double>("sigmaIetaMin"); 
  double sigmaIetaMax = parameters_.getParameter<double>("sigmaIetaMax"); 
  int sigmaIetaBin = parameters_.getParameter<int>("sigmaIetaBin");

  double eOverPMin = parameters_.getParameter<double>("eOverPMin"); 
  double eOverPMax = parameters_.getParameter<double>("eOverPMax"); 
  int eOverPBin = parameters_.getParameter<int>("eOverPBin");

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

    //Invariant mass plots

    currentFolder_.str("");
    currentFolder_ << "Egamma/PhotonAnalyzer/InvMass";
    dbe_->setCurrentFolder(currentFolder_.str());
    

    h_invMassTwoWithTracks_= dbe_->book1D("invMassTwoWithTracks"," Two photon invariant mass: Both have tracks ",etBin,etMin,etMax);
    h_invMassOneWithTracks_= dbe_->book1D("invMassOneWithTracks"," Two photon invariant mass: Only one has tracks ",etBin,etMin,etMax);
    h_invMassZeroWithTracks_= dbe_->book1D("invMassZeroWithTracks"," Two photon invariant mass: Neither have tracks ",etBin,etMin,etMax);
    h_invMassAllPhotons_= dbe_->book1D("invMassAllIsolatedPhotons"," Two photon invariant mass: All isolated photons",etBin,etMin,etMax);





    //Efficiency histograms

    currentFolder_.str("");
    currentFolder_ << "Egamma/PhotonAnalyzer/Efficiencies";
    dbe_->setCurrentFolder(currentFolder_.str());

    h_phoEta_Loose_ = dbe_->book1D("phoEtaLoose"," Loose Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEta_Tight_ = dbe_->book1D("phoEtaTight"," Tight Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEta_HLT_ = dbe_->book1D("phoEtaHLT"," Unfiltered Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEt_Loose_ = dbe_->book1D("phoEtLoose"," Loose Photon Transverse Energy ", etBin,etMin,etMax);
    h_phoEt_Tight_ = dbe_->book1D("phoEtTight"," Tight Photon Transverse Energy ", etBin,etMin,etMax);
    h_phoEt_HLT_ = dbe_->book1D("phoEtHLT"," Unfiltered Photon Transverse Energy ", etBin,etMin,etMax);

    h_convEta_Loose_ = dbe_->book1D("convEtaLoose"," Converted Loose Photon Eta ",etaBin,etaMin, etaMax) ;
    h_convEta_Tight_ = dbe_->book1D("convEtaTight"," Converted Tight Photon Eta ",etaBin,etaMin, etaMax) ;
    h_convEt_Loose_ = dbe_->book1D("convEtLoose"," Converted Loose Photon Transverse Energy ", etBin,etMin,etMax);
    h_convEt_Tight_ = dbe_->book1D("convEtTight"," Converted Tight Photon Transverse Energy ", etBin,etMin,etMax);

    h_phoEta_Vertex_ = dbe_->book1D("phoEtaVertex"," Converted Photons before valid vertex cut: Eta ",etaBin,etaMin, etaMax) ;

    //Triggers passed
    
    h_filters_ = dbe_->book1D("Filters","Filters Passed;;Fraction of Photons Passing",11,0,11);


    for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
      
      // Photon histograms

      for(uint type=0;type!=types.size();++type){ //looping over isolation type
	
	currentFolder_.str("");
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
	dbe_->setCurrentFolder(currentFolder_.str());

	for(uint part=0;part!=parts.size();++part){ //looping over different parts of the ecal

	  h_phoE_part_.push_back(dbe_->book1D("phoE"+parts[part],types[type]+" Photon Energy: "+parts[part]+";E (GeV)", eBin,eMin,eMax));
	  h_phoEt_part_.push_back(dbe_->book1D("phoEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part]+";Et (GeV)", etBin,etMin,etMax));
	  h_r9_part_.push_back(dbe_->book1D("r9"+parts[part],types[type]+" Photon r9: "+parts[part]+";R9",r9Bin,r9Min, r9Max));
	  h_hOverE_part_.push_back(dbe_->book1D("hOverE"+parts[part],types[type]+" Photon H/E: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_h1OverE_part_.push_back(dbe_->book1D("h1OverE"+parts[part],types[type]+" Photon H/E for Depth 1: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_h2OverE_part_.push_back(dbe_->book1D("h2OverE"+parts[part],types[type]+" Photon H/E for Depth 2: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_phoSigmaIetaIeta_part_.push_back(dbe_->book1D("phoSigmaIetaIeta"+parts[part],types[type]+" Photon #sigmai#etai#eta: "+parts[part]+";#sigmai#etai#eta",sigmaIetaBin,sigmaIetaMin,sigmaIetaMax)) ;
	  h_nPho_part_.push_back(dbe_->book1D("nPho"+parts[part],"Number of "+types[type]+" Photons per Event: "+parts[part]+";# #gamma", numberBin,numberMin,numberMax));
	  p_ecalSumVsEt_part_.push_back(dbe_->bookProfile("ecalSumVsEt"+parts[part],"Avg Ecal Sum in the Iso Cone vs.  E_{T}: "+parts[part]+";E_{T};E (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax,""));
	  p_hcalSumVsEt_part_.push_back(dbe_->bookProfile("hcalSumVsEt"+parts[part],"Avg Hcal Sum in the Iso Cone vs.  E_{T}: "+parts[part]+";E_{T};E (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax,""));



	}//end loop over different parts of the ecal


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
	h_phoSigmaIetaIeta_isol_.push_back(h_phoSigmaIetaIeta_part_);
	h_phoSigmaIetaIeta_part_.clear();
	h_nPho_isol_.push_back(h_nPho_part_);
	h_nPho_part_.clear();


	p_ecalSumVsEt_isol_.push_back(p_ecalSumVsEt_part_);
	p_hcalSumVsEt_isol_.push_back(p_hcalSumVsEt_part_);
	p_ecalSumVsEt_part_.clear();
	p_hcalSumVsEt_part_.clear();


	h_phoEta_isol_.push_back(dbe_->book1D("phoEta",types[type]+" Photon Eta;#eta ",etaBin,etaMin, etaMax)) ;
	h_phoPhi_isol_.push_back(dbe_->book1D("phoPhi",types[type]+" Photon Phi;#phi ",phiBin,phiMin,phiMax)) ;

	h_phoEta_BadChannels_isol_.push_back(dbe_->book1D("phoEtaBadChannels",types[type]+"Photons with bad channels: Eta",etaBin,etaMin, etaMax)) ;
	h_phoEt_BadChannels_isol_.push_back(dbe_->book1D("phoEtBadChannels",types[type]+"Photons with bad channels: Et ", etBin,etMin,etMax));
	h_phoPhi_BadChannels_isol_.push_back(dbe_->book1D("phoPhiBadChannels",types[type]+"Photons with bad channels: Phi ", phiBin,phiMin, phiMax));


	h_scEta_isol_.push_back(dbe_->book1D("scEta",types[type]+" SuperCluster Eta;#eta ",etaBin,etaMin, etaMax)) ;
	h_scPhi_isol_.push_back(dbe_->book1D("scPhi",types[type]+" SuperCluster Phi;#phi ",phiBin,phiMin,phiMax)) ;


	h_r9VsEt_isol_.push_back(dbe_->book2D("r9VsEt2D",types[type]+" Photon r9 vs. Transverse Energy;Et (GeV);R9",etBin/2,etMin,etMax,r9Bin/2,r9Min,r9Max));
	h_r9VsEta_isol_.push_back(dbe_->book2D("r9VsEta2D",types[type]+" Photon r9 vs. #eta;#eta;R9",etaBin/2,etaMin,etaMax,r9Bin/2,r9Min,r9Max));

	if(type==0){
	  h_e1x5VsEt_isol_.push_back(dbe_->book2D("e1x5VsEt2D",types[type]+" Photon e1x5 vs. Transverse Energy;Et (GeV);E1X5",etBin/2,etMin,etMax,etBin/2,etMin,etMax));
	  h_e1x5VsEta_isol_.push_back(dbe_->book2D("e1x5VsEta2D",types[type]+" Photon e1x5 vs. #eta;#eta;E1X5",etaBin/2,etaMin,etaMax,etBin/2,etMin,etMax));
	  
	  h_e2x5VsEt_isol_.push_back(dbe_->book2D("e2x5VsEt2D",types[type]+" Photon e2x5 vs. Transverse Energy;Et (GeV);E2X5",etBin/2,etMin,etMax,etBin/2,etMin,etMax));
	  h_e2x5VsEta_isol_.push_back(dbe_->book2D("e2x5VsEta2D",types[type]+" Photon e2x5 vs. #eta;#eta;E2X5",etaBin/2,etaMin,etaMax,etBin/2,etMin,etMax));

	  h_maxEXtalOver3x3VsEt_isol_.push_back(dbe_->book2D("maxEXtalOver3x3VsEt2D",types[type]+" Photon MaxE xtal/3x3 vs. Transverse Energy;Et (GeV); Max Xtal E/e3x3",etBin/2,etMin,etMax,etBin/2,etMin,etMax));
	  h_maxEXtalOver3x3VsEta_isol_.push_back(dbe_->book2D("maxEXtalOver3x3VsEta2D",types[type]+" Photon MaxE xtal/3x3 vs. #eta;#eta; Max Xtal E/e3x3",etaBin/2,etaMin,etaMax,etBin/2,etMin,etMax));
	  
	  h_r1x5VsEt_isol_.push_back(dbe_->book2D("r1x5VsEt2D",types[type]+" Photon r1x5 vs. Transverse Energy;Et (GeV);R1X5",etBin/2,etMin,etMax,r9Bin/2,r9Min,r9Max));
	  h_r1x5VsEta_isol_.push_back(dbe_->book2D("r1x5VsEta2D",types[type]+" Photon r1x5 vs. #eta;#eta;R1X5",etaBin/2,etaMin,etaMax,r9Bin/2,r9Min,r9Max));
	  
	  h_r2x5VsEt_isol_.push_back(dbe_->book2D("r2x5VsEt2D",types[type]+" Photon r2x5 vs. Transverse Energy;Et (GeV);R2X5",etBin/2,etMin,etMax,r9Bin/2,r9Min,r9Max));
	  h_r2x5VsEta_isol_.push_back(dbe_->book2D("r2x5VsEta2D",types[type]+" Photon r2x5 vs. #eta;#eta;R2X5",etaBin/2,etaMin,etaMax,r9Bin/2,r9Min,r9Max));
	}
	

	h_sigmaIetaIetaVsEta_isol_.push_back(dbe_->book2D("sigmaIetaIetaVsEta2D",types[type]+" Photon #sigmai#etai#eta vs. #eta;#eta;#sigmai#etai#eta",etaBin/2,etaMin,etaMax,sigmaIetaBin/2,sigmaIetaMin,sigmaIetaMax));

	p_r9VsEt_isol_.push_back(dbe_->bookProfile("r9VsEt",types[type]+" Photon r9 vs. Transverse Energy;Et (GeV);R9",etBin,etMin,etMax,r9Bin,r9Min,r9Max,""));
	p_r9VsEta_isol_.push_back(dbe_->bookProfile("r9VsEta",types[type]+" Photon r9 vs. #eta;#eta;R9",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max,""));
	
	p_e1x5VsEt_isol_.push_back(dbe_->bookProfile("e1x5VsEt",types[type]+" Photon e1x5 vs. Transverse Energy;Et (GeV);E1X5",etBin,etMin,etMax,etBin,etMin,etMax,""));
	p_e1x5VsEta_isol_.push_back(dbe_->bookProfile("e1x5VsEta",types[type]+" Photon e1x5 vs. #eta;#eta;E1X5",etaBin,etaMin,etaMax,etBin,etMin,etMax,""));
	
	p_e2x5VsEt_isol_.push_back(dbe_->bookProfile("e2x5VsEt",types[type]+" Photon e2x5 vs. Transverse Energy;Et (GeV);E2X5",etBin,etMin,etMax,etBin,etMin,etMax,""));
	p_e2x5VsEta_isol_.push_back(dbe_->bookProfile("e2x5VsEta",types[type]+" Photon e2x5 vs. #eta;#eta;E2X5",etaBin,etaMin,etaMax,etBin,etMin,etMax,""));
	
	p_maxEXtalOver3x3VsEt_isol_.push_back(dbe_->bookProfile("maxEXtalOver3x3VsEt",types[type]+" Photon max Xtal E/e3x3 vs. Transverse Energy;Et (GeV);maxE/e3x3",etBin,etMin,etMax,etBin,etMin,etMax,""));
	p_maxEXtalOver3x3VsEta_isol_.push_back(dbe_->bookProfile("maxEXtalOver3x3VsEta",types[type]+" Photon max Xtal E/e3x3 vs. #eta;#eta;maxE/3x3",etaBin,etaMin,etaMax,etBin,etMin,etMax,""));

	p_r1x5VsEt_isol_.push_back(dbe_->bookProfile("r1x5VsEt",types[type]+" Photon r1x5 vs. Transverse Energy;Et (GeV);R1X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max,""));
	p_r1x5VsEta_isol_.push_back(dbe_->bookProfile("r1x5VsEta",types[type]+" Photon r1x5 vs. #eta;#eta;R1X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max,""));
	
	p_r2x5VsEt_isol_.push_back(dbe_->bookProfile("r2x5VsEt",types[type]+" Photon r2x5 vs. Transverse Energy;Et (GeV);R2X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max,""));
	p_r2x5VsEta_isol_.push_back(dbe_->bookProfile("r2x5VsEta",types[type]+" Photon r2x5 vs. #eta;#eta;R2X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max,""));
	
	
	p_sigmaIetaIetaVsEta_isol_.push_back(dbe_->bookProfile("sigmaIetaIetaVsEta",types[type]+" Photon #sigmai#etai#eta vs. #eta;#eta;#sigmai#etai#eta",etaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax,""));
	
	p_hOverEVsEta_isol_.push_back(dbe_->bookProfile("hOverEVsEta",types[type]+" Photon H/E vs Eta;#eta;H/E",etaBin,etaMin,etaMax,hOverEBin,hOverEMin,hOverEMax,""));
	p_hOverEVsEt_isol_.push_back(dbe_->bookProfile("hOverEVsEt",types[type]+" Photon H/E vs Et;Et (GeV);H/E",etBin,etMin,etMax,hOverEBin,hOverEMin,hOverEMax,""));
	      



	// Isolation Variable infos
	if(type==0){
	  h_nTrackIsolSolidVsEta_isol_.push_back(dbe_->book2D("nIsoTracksSolidVsEta2D","Avg Number Of Tracks in the Solid Iso Cone vs.  #eta;#eta;# tracks",etaBin/2,etaMin, etaMax,numberBin/2,numberMin,numberMax));
	  h_trackPtSumSolidVsEta_isol_.push_back(dbe_->book2D("isoPtSumSolidVsEta2D","Avg Tracks Pt Sum in the Solid Iso Cone",etaBin/2,etaMin, etaMax,sumBin/2,sumMin,sumMax));
	  h_nTrackIsolHollowVsEta_isol_.push_back(dbe_->book2D("nIsoTracksHollowVsEta2D","Avg Number Of Tracks in the Hollow Iso Cone vs.  #eta;#eta;# tracks",etaBin/2,etaMin, etaMax,numberBin/2,numberMin,numberMax));
	  h_trackPtSumHollowVsEta_isol_.push_back(dbe_->book2D("isoPtSumHollowVsEta2D","Avg Tracks Pt Sum in the Hollow Iso Cone",etaBin/2,etaMin, etaMax,sumBin/2,sumMin,sumMax));

	  h_nTrackIsolSolidVsEt_isol_.push_back(dbe_->book2D("nIsoTracksSolidVsEt2D","Avg Number Of Tracks in the Solid Iso Cone vs.  E_{T};E_{T};# tracks",etBin/2,etMin, etMax,numberBin/2,numberMin,numberMax));
	  h_trackPtSumSolidVsEt_isol_.push_back(dbe_->book2D("isoPtSumSolidVsEt2D","Avg Tracks Pt Sum in the Solid Iso Cone",etBin/2,etMin, etMax,sumBin/2,sumMin,sumMax));
	  h_nTrackIsolHollowVsEt_isol_.push_back(dbe_->book2D("nIsoTracksHollowVsEt2D","Avg Number Of Tracks in the Hollow Iso Cone vs.  E_{T};E_{T};# tracks",etBin/2,etMin, etMax,numberBin/2,numberMin,numberMax));
	  h_trackPtSumHollowVsEt_isol_.push_back(dbe_->book2D("isoPtSumHollowVsEt2D","Avg Tracks Pt Sum in the Hollow Iso Cone",etBin/2,etMin, etMax,sumBin/2,sumMin,sumMax));
	}

	h_ecalSumVsEta_isol_.push_back(dbe_->book2D("ecalSumVsEta2D","Avg Ecal Sum in the Iso Cone",etaBin/2,etaMin, etaMax,sumBin/2,sumMin,sumMax));
	h_hcalSumVsEta_isol_.push_back(dbe_->book2D("hcalSumVsEta2D","Avg Hcal Sum in the Iso Cone",etaBin/2,etaMin, etaMax,sumBin/2,sumMin,sumMax));
	h_ecalSumVsEt_isol_.push_back(dbe_->book2D("ecalSumVsEt2D","Avg Ecal Sum in the Iso Cone",etBin/2,etMin, etMax,sumBin/2,sumMin,sumMax));
	h_hcalSumVsEt_isol_.push_back(dbe_->book2D("hcalSumVsEt2D","Avg Hcal Sum in the Iso Cone",etBin/2,etMin, etMax,sumBin/2,sumMin,sumMax));
		  

	h_nTrackIsolSolid_isol_.push_back(dbe_->book1D("nIsoTracksSolid","Avg Number Of Tracks in the Solid Iso Cone;# tracks",numberBin,numberMin,numberMax));
	h_trackPtSumSolid_isol_.push_back(dbe_->book1D("isoPtSumSolid","Avg Tracks Pt Sum in the Solid Iso Cone;Pt (GeV)",sumBin,sumMin,sumMax));
	h_nTrackIsolHollow_isol_.push_back(dbe_->book1D("nIsoTracksHollow","Avg Number Of Tracks in the Hollow Iso Cone;# tracks",numberBin,numberMin,numberMax));
	h_trackPtSumHollow_isol_.push_back(dbe_->book1D("isoPtSumHollow","Avg Tracks Pt Sum in the Hollow Iso Cone;Pt (GeV)",sumBin,sumMin,sumMax));
	h_ecalSum_isol_.push_back(dbe_->book1D("ecalSum","Avg Ecal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax));
	h_hcalSum_isol_.push_back(dbe_->book1D("hcalSum","Avg Hcal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax));

	p_nTrackIsolSolidVsEta_isol_.push_back(dbe_->bookProfile("nIsoTracksSolidVsEta","Avg Number Of Tracks in the Solid Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax,""));
	p_trackPtSumSolidVsEta_isol_.push_back(dbe_->bookProfile("isoPtSumSolidVsEta","Avg Tracks Pt Sum in the Solid Iso Cone vs.  #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax,""));
	p_nTrackIsolHollowVsEta_isol_.push_back(dbe_->bookProfile("nIsoTracksHollowVsEta","Avg Number Of Tracks in the Hollow Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax,""));
	p_trackPtSumHollowVsEta_isol_.push_back(dbe_->bookProfile("isoPtSumHollowVsEta","Avg Tracks Pt Sum in the Hollow Iso Cone vs.  #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax,""));
	p_ecalSumVsEta_isol_.push_back(dbe_->bookProfile("ecalSumVsEta","Avg Ecal Sum in the Iso Cone vs.  #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax,""));
	p_hcalSumVsEta_isol_.push_back(dbe_->bookProfile("hcalSumVsEta","Avg Hcal Sum in the Iso Cone vs.  #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax,""));
	
	
	p_nTrackIsolSolidVsEt_isol_.push_back(dbe_->bookProfile("nIsoTracksSolidVsEt","Avg Number Of Tracks in the Solid Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax,""));
	p_trackPtSumSolidVsEt_isol_.push_back(dbe_->bookProfile("isoPtSumSolidVsEt","Avg Tracks Pt Sum in the Solid Iso Cone vs.  E_{T};E_{T};Pt (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax,""));
	p_nTrackIsolHollowVsEt_isol_.push_back(dbe_->bookProfile("nIsoTracksHollowVsEt","Avg Number Of Tracks in the Hollow Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax,""));
	p_trackPtSumHollowVsEt_isol_.push_back(dbe_->bookProfile("isoPtSumHollowVsEt","Avg Tracks Pt Sum in the Hollow Iso Cone vs.  E_{T};E_{T};Pt (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax,""));

	





      }//end loop over isolation type


      h_phoE_.push_back(h_phoE_isol_);
      h_phoE_isol_.clear();
      h_phoEt_.push_back(h_phoEt_isol_);
      h_phoEt_isol_.clear();

      h_phoEt_BadChannels_.push_back(h_phoEt_BadChannels_isol_);
      h_phoEt_BadChannels_isol_.clear();
      h_phoEta_BadChannels_.push_back(h_phoEta_BadChannels_isol_);
      h_phoEta_BadChannels_isol_.clear();
      h_phoPhi_BadChannels_.push_back(h_phoPhi_BadChannels_isol_);
      h_phoPhi_BadChannels_isol_.clear();

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
      
            
      h_phoEta_.push_back(h_phoEta_isol_);
      h_phoEta_isol_.clear();
      h_phoPhi_.push_back(h_phoPhi_isol_);
      h_phoPhi_isol_.clear();

      h_scEta_.push_back(h_scEta_isol_);
      h_scEta_isol_.clear();
      h_scPhi_.push_back(h_scPhi_isol_);
      h_scPhi_isol_.clear();

      h_r9VsEt_.push_back(h_r9VsEt_isol_);
      h_r9VsEt_isol_.clear();
      h_r9VsEta_.push_back(h_r9VsEta_isol_);
      h_r9VsEta_isol_.clear();

      h_e1x5VsEt_.push_back(h_e1x5VsEt_isol_);
      h_e1x5VsEt_isol_.clear();
      h_e1x5VsEta_.push_back(h_e1x5VsEta_isol_);
      h_e1x5VsEta_isol_.clear();

      h_e2x5VsEt_.push_back(h_e2x5VsEt_isol_);
      h_e2x5VsEt_isol_.clear();
      h_e2x5VsEta_.push_back(h_e2x5VsEta_isol_);
      h_e2x5VsEta_isol_.clear();

      h_maxEXtalOver3x3VsEt_.push_back(h_maxEXtalOver3x3VsEt_isol_);
      h_maxEXtalOver3x3VsEt_isol_.clear();
      h_maxEXtalOver3x3VsEta_.push_back(h_maxEXtalOver3x3VsEta_isol_);
      h_maxEXtalOver3x3VsEta_isol_.clear();


      h_r1x5VsEt_.push_back(h_r1x5VsEt_isol_);
      h_r1x5VsEt_isol_.clear();
      h_r1x5VsEta_.push_back(h_r1x5VsEta_isol_);
      h_r1x5VsEta_isol_.clear();

      h_r2x5VsEt_.push_back(h_r2x5VsEt_isol_);
      h_r2x5VsEt_isol_.clear();
      h_r2x5VsEta_.push_back(h_r2x5VsEta_isol_);
      h_r2x5VsEta_isol_.clear();

 
      h_phoSigmaIetaIeta_.push_back(h_phoSigmaIetaIeta_isol_);
      h_phoSigmaIetaIeta_isol_.clear();
  
      h_sigmaIetaIetaVsEta_.push_back(h_sigmaIetaIetaVsEta_isol_);
      h_sigmaIetaIetaVsEta_isol_.clear();


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


      p_maxEXtalOver3x3VsEt_.push_back(p_maxEXtalOver3x3VsEt_isol_);
      p_maxEXtalOver3x3VsEt_isol_.clear();
      p_maxEXtalOver3x3VsEta_.push_back(p_maxEXtalOver3x3VsEta_isol_);
      p_maxEXtalOver3x3VsEta_isol_.clear();
      
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
      

      p_hOverEVsEta_.push_back(p_hOverEVsEta_isol_);
      p_hOverEVsEta_isol_.clear();
      p_hOverEVsEt_.push_back(p_hOverEVsEt_isol_);
      p_hOverEVsEt_isol_.clear();


      h_nTrackIsolSolidVsEta_.push_back(h_nTrackIsolSolidVsEta_isol_);
      h_trackPtSumSolidVsEta_.push_back(h_trackPtSumSolidVsEta_isol_);
      h_nTrackIsolHollowVsEta_.push_back(h_nTrackIsolHollowVsEta_isol_);
      h_trackPtSumHollowVsEta_.push_back(h_trackPtSumHollowVsEta_isol_);
      h_ecalSumVsEta_.push_back(h_ecalSumVsEta_isol_);
      h_hcalSumVsEta_.push_back(h_hcalSumVsEta_isol_);

    
      h_nTrackIsolSolidVsEta_isol_.clear();
      h_trackPtSumSolidVsEta_isol_.clear();
      h_nTrackIsolHollowVsEta_isol_.clear();
      h_trackPtSumHollowVsEta_isol_.clear();
      h_ecalSumVsEta_isol_.clear();
      h_hcalSumVsEta_isol_.clear();


      h_nTrackIsolSolidVsEt_.push_back(h_nTrackIsolSolidVsEt_isol_);
      h_trackPtSumSolidVsEt_.push_back(h_trackPtSumSolidVsEt_isol_);
      h_nTrackIsolHollowVsEt_.push_back(h_nTrackIsolHollowVsEt_isol_);
      h_trackPtSumHollowVsEt_.push_back(h_trackPtSumHollowVsEt_isol_);
      h_ecalSumVsEt_.push_back(h_ecalSumVsEt_isol_);
      h_hcalSumVsEt_.push_back(h_hcalSumVsEt_isol_);

    
      h_nTrackIsolSolidVsEt_isol_.clear();
      h_trackPtSumSolidVsEt_isol_.clear();
      h_nTrackIsolHollowVsEt_isol_.clear();
      h_trackPtSumHollowVsEt_isol_.clear();
      h_ecalSumVsEt_isol_.clear();
      h_hcalSumVsEt_isol_.clear();


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
      
      
      p_nTrackIsolSolidVsEt_isol_.clear();
      p_trackPtSumSolidVsEt_isol_.clear();
      p_nTrackIsolHollowVsEt_isol_.clear();
      p_trackPtSumHollowVsEt_isol_.clear();

      p_nTrackIsolSolidVsEta_isol_.clear();
      p_trackPtSumSolidVsEta_isol_.clear();
      p_nTrackIsolHollowVsEta_isol_.clear();
      p_trackPtSumHollowVsEta_isol_.clear();


      p_ecalSumVsEt_.push_back(p_ecalSumVsEt_isol_);
      p_hcalSumVsEt_.push_back(p_hcalSumVsEt_isol_);
      p_ecalSumVsEta_isol_.clear();
      p_hcalSumVsEta_isol_.clear();
      
      
      
      //conversion plots
      
      for(uint type=0;type!=types.size();++type){ //looping over isolation type
	
	currentFolder_.str("");	
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/Conversions";
	dbe_->setCurrentFolder(currentFolder_.str());

	for(uint part=0;part!=parts.size();++part){ //loop over different parts of the ecal

	  h_phoConvE_part_.push_back(dbe_->book1D("phoConvE"+parts[part],types[type]+" Photon Energy: "+parts[part]+";E (GeV)", eBin,eMin,eMax));
	  h_phoConvEt_part_.push_back(dbe_->book1D("phoConvEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part]+";Et (GeV)", etBin,etMin,etMax));
	  h_phoConvR9_part_.push_back(dbe_->book1D("phoConvR9"+parts[part],types[type]+" Photon r9: "+parts[part]+";R9",r9Bin,r9Min, r9Max));
	  h_nConv_part_.push_back(dbe_->book1D("nConv"+parts[part],"Number Of Conversions per Event:  "+parts[part]+";# conversions" ,numberBin,numberMin,numberMax));

	  h_eOverPTracks_part_.push_back(dbe_->book1D("eOverPTracks"+parts[part],"E/P of Conversions: "+parts[part]+";E/P" ,eOverPBin,eOverPMin,eOverPMax));
	  h_pOverETracks_part_.push_back(dbe_->book1D("pOverETracks"+parts[part],"P/E of Conversions: "+parts[part]+";P/E" ,eOverPBin,eOverPMin,eOverPMax));
	  	  
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
	h_pOverETracks_isol_.push_back(h_pOverETracks_part_);
	h_pOverETracks_part_.clear();

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

	h_phoConvEtaForEfficiency_isol_.push_back(dbe_->book1D("phoConvEtaForEfficiency",types[type]+" Converted Photon Eta;#eta ",etaBin,etaMin, etaMax)) ;
	h_phoConvPhiForEfficiency_isol_.push_back(dbe_->book1D("phoConvPhiForEfficiency",types[type]+" Converted Photon Phi;#phi ",phiBin,phiMin,phiMax)) ;

	h_convVtxRvsZ_isol_.push_back(dbe_->book2D("convVtxRvsZ",types[type]+" Photon Reco conversion vtx position;Z (cm);R (cm)",zBin,zMin,zMax,rBin,rMin,rMax));
	h_convVtxZ_isol_.push_back(dbe_->book1D("convVtxZ",types[type]+" Photon Reco conversion vtx position: #eta > 1.5;Z (cm)",zBin,zMin,zMax));
	h_convVtxR_isol_.push_back(dbe_->book1D("convVtxR",types[type]+" Photon Reco conversion vtx position: #eta < 1;R (cm)",rBin,rMin,rMax));
	h_convVtxYvsX_isol_.push_back(dbe_->book2D("convVtxYvsX",types[type]+" Photon Reco conversion vtx position: #eta < 1;X (cm);Y (cm)",xBin,xMin,xMax,yBin,yMin,yMax));

	p_nHitsVsEta_isol_.push_back(dbe_->bookProfile("nHitsVsEta",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta;#eta;# hits",etaBin,etaMin, etaMax,etaBin,0, 16,""));
	p_tkChi2VsEta_isol_.push_back(dbe_->bookProfile("tkChi2VsEta",types[type]+" Photons: Tracks from conversions: #chi^{2} vs Eta;#eta;#chi^{2}",etaBin,etaMin, etaMax,100, 0., 20.0,""));
	p_dCotTracksVsEta_isol_.push_back(dbe_->bookProfile("dCotTracksVsEta",types[type]+" #delta cotg(#Theta) of Conversion Tracks vs Eta;#eta;#delta cotg(#Theta)",etaBin,etaMin, etaMax,dEtaTracksBin,dEtaTracksMin,dEtaTracksMax,""));
	h_tkChi2_isol_.push_back(dbe_->book1D("tkChi2",types[type]+" Photons: Tracks from conversions: #chi^{2} of all tracks;#chi^{2}", 100, 0., 20.0));
	h_vertexChi2_isol_.push_back(dbe_->book1D("vertexChi2",types[type]+" Photons: Tracks from conversions: Vertex fitting #chi^{2};#chi^{2}", 100, 0., 1.0));

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
      h_pOverETracks_.push_back(h_pOverETracks_isol_);
      h_pOverETracks_isol_.clear();
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

      h_phoConvEtaForEfficiency_.push_back(h_phoConvEtaForEfficiency_isol_);
      h_phoConvEtaForEfficiency_isol_.clear();
      h_phoConvPhiForEfficiency_.push_back(h_phoConvPhiForEfficiency_isol_);
      h_phoConvPhiForEfficiency_isol_.clear();

      h_convVtxRvsZ_.push_back(h_convVtxRvsZ_isol_);
      h_convVtxRvsZ_isol_.clear();
      h_convVtxR_.push_back(h_convVtxR_isol_);
      h_convVtxR_isol_.clear();
      h_convVtxZ_.push_back(h_convVtxZ_isol_);
      h_convVtxZ_isol_.clear();
      h_convVtxYvsX_.push_back(h_convVtxYvsX_isol_);
      h_convVtxYvsX_isol_.clear();

      p_nHitsVsEta_.push_back(p_nHitsVsEta_isol_);
      p_nHitsVsEta_isol_.clear(); 
      p_tkChi2VsEta_.push_back(p_tkChi2VsEta_isol_);
      p_tkChi2VsEta_isol_.clear(); 
      p_dCotTracksVsEta_.push_back(p_dCotTracksVsEta_isol_);
      p_dCotTracksVsEta_isol_.clear(); 
      h_tkChi2_.push_back(h_tkChi2_isol_);
      h_tkChi2_isol_.clear();
      h_vertexChi2_.push_back(h_vertexChi2_isol_);
      h_vertexChi2_isol_.clear();




    }



  }


}
 
 




void PhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  using namespace edm;
 
  if (nEvt_% prescaleFactor_ ) return; 
  nEvt_++;  
  LogInfo("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";

  // Get the trigger results
  bool validTriggerEvent=true;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  trigger::TriggerEvent triggerEvent;
  e.getByLabel(triggerEvent_,triggerEventHandle);
  if(!triggerEventHandle.isValid()) {
    edm::LogInfo("PhotonAnalyzer") << "Error! Can't get the product "<< triggerEvent_.label() << endl;
    validTriggerEvent=false;
  }
  if(validTriggerEvent) triggerEvent = *(triggerEventHandle.product());

  // Get the reconstructed photons
  bool validPhotons=true;
  Handle<reco::PhotonCollection> photonHandle; 
  reco::PhotonCollection photonCollection;
  e.getByLabel(photonProducer_, photonCollection_ , photonHandle);
  if ( !photonHandle.isValid()) {
    edm::LogInfo("PhotonAnalyzer") << "Error! Can't get the product "<< photonCollection_ << endl;
    validPhotons=false;
  }
  if(validPhotons) photonCollection = *(photonHandle.product());

  // Get the PhotonId objects
  bool validloosePhotonID=true;
  Handle<edm::ValueMap<bool> > loosePhotonFlag;
  edm::ValueMap<bool> loosePhotonID;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDLoose", loosePhotonFlag);
  if ( !loosePhotonFlag.isValid()) {
    edm::LogInfo("PhotonAnalyzer") << "Error! Can't get the product "<< "PhotonCutBasedIDLoose" << endl;
    validloosePhotonID=false;
  }
  if (validloosePhotonID) loosePhotonID = *(loosePhotonFlag.product());

  bool validtightPhotonID=true;
  Handle<edm::ValueMap<bool> > tightPhotonFlag;
  edm::ValueMap<bool> tightPhotonID;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDTight", tightPhotonFlag);
  if ( !tightPhotonFlag.isValid()) {
    edm::LogInfo("PhotonAnalyzer") << "Error! Can't get the product "<< "PhotonCutBasedIDTight" << endl;
    validtightPhotonID=false;
  }
  if (validtightPhotonID) tightPhotonID = *(tightPhotonFlag.product());



  // Create array to hold #photons/event information
  int nPho[100][3][3];

  for (int cut=0; cut!=100; ++cut){
    for (int type=0; type!=3; ++type){
      for (int part=0; part!=3; ++part){
	nPho[cut][type][part] = 0;
      }
    }
  }
  // Create array to hold #conversions/event information
  int nConv[100][3][3];

  for (int cut=0; cut!=100; ++cut){
    for (int type=0; type!=3; ++type){
      for (int part=0; part!=3; ++part){
	nConv[cut][type][part] = 0;
      }
    }
  }


 
  //Prepare list of photon-related HLT filter names

  std::vector<int> Keys;
 

  TH1 *filters = h_filters_->getTH1();

  if(nEvt_ == 1) {
    triggerLabels.push_back("hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilterESet70");
    triggerLabels.push_back("hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter");
    triggerLabels.push_back("hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter");
  

    for (uint bin =1;bin!=triggerLabels.size()+1;bin++){
      filters->GetXaxis()->SetBinLabel(bin,triggerLabels[bin-1].c_str());
    }

  }
  

  for(uint filterIndex=0;filterIndex<triggerEvent.sizeFilters();++filterIndex){  //loop over all trigger filters in event (i.e. filters passed)

    string label = triggerEvent.filterTag(filterIndex).label();

    if(label.find( "Photon" ) != std::string::npos ) {  //get photon-related filters and fill histo
     
      for (uint bin =0;bin!=triggerLabels.size();++bin){
	if (label==triggerLabels[bin]) h_filters_->Fill(bin);
      }
      

      for(uint filterKeyIndex=0;filterKeyIndex<triggerEvent.filterKeys(filterIndex).size();++filterKeyIndex){  //loop over keys to objects passing this filter
	Keys.push_back(triggerEvent.filterKeys(filterIndex)[filterKeyIndex]);  //add keys to a vector for later reference	
      }  
      
    }
    
  }

  sort(Keys.begin(),Keys.end());  //sort Keys vector in ascending order

  for(uint i=0;i<Keys.size();){  //erases duplicate entries from the vector
    if(Keys[i]==Keys[i+1] && i!=Keys.size()-1) Keys.erase(Keys.begin()+i+1);
    else ++i;
  }

  //We now have a vector of unique keys to TriggerObjects passing a photon-related filter
 



  int photonCounter = 0;

  // Loop over all photons in event
  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {



    //for reconstruction efficiency plots
    h_phoEta_HLT_->Fill( (*iPho).eta() );
    h_phoEt_HLT_->Fill( (*iPho).et() ); 

    double deltaR=1000.;
    double deltaRMin=1000.;
    double deltaRMax=0.05;
    
    for (vector<int>::const_iterator objectKey=Keys.begin();objectKey!=Keys.end();objectKey++){  //loop over keys to objects that fired photon triggers

      deltaR = reco::deltaR(triggerEvent.getObjects()[(*objectKey)].eta(),triggerEvent.getObjects()[(*objectKey)].phi(),(*iPho).superCluster()->eta(),(*iPho).superCluster()->phi());
      if(deltaR < deltaRMin) deltaRMin = deltaR;
      
    }

    if(deltaRMin > deltaRMax) {  //photon fails delta R cut

      if(useTriggerFiltering_) continue;  //throw away photons that haven't passed any photon filters
    }


    if ((*iPho).et()  < minPhoEtCut_) continue;

    nEntry_++;

    edm::Ref<reco::PhotonCollection> photonref(photonHandle, photonCounter);
    photonCounter++;
    
    bool isLoosePhoton(false), isTightPhoton(false);
    if ( !isHeavyIon_ ) {
       isLoosePhoton = (loosePhotonID)[photonref];
       isTightPhoton = (tightPhotonID)[photonref];
    }
    

    //find which part of the Ecal contains the photon

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

    int part = 0;
    if ( phoIsInBarrel ) part = 1;
    if ( phoIsInEndcap ) part = 2;

    /////  From 30X on, Photons are already pre-selected at reconstruction level with a looseEM isolation
    bool isIsolated=false;
    if ( isolationStrength_ == 0)  isIsolated = isLoosePhoton;
    if ( isolationStrength_ == 1)  isIsolated = isTightPhoton; 

    int type=0;
    if ( isIsolated ) type=1;
    if ( !isIsolated ) type=2;


    //get rechit collection containing this photon
    bool validEcalRecHits=true;
    edm::Handle<EcalRecHitCollection>   ecalRecHitHandle;
    EcalRecHitCollection ecalRecHitCollection;
    if ( phoIsInBarrel ) {
      // Get handle to rec hits ecal barrel 
      e.getByLabel(barrelRecHitProducer_, barrelRecHitCollection_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError("PhotonAnalyzer") << "Error! Can't get the product "<<barrelRecHitProducer_;
	validEcalRecHits=false; 
	}

    } else if ( phoIsInEndcap ) {    
      // Get handle to rec hits ecal encap 
      e.getByLabel(endcapRecHitProducer_, endcapRecHitCollection_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError("PhotonAnalyzer") << "Error! Can't get the product "<<endcapRecHitProducer_;
	validEcalRecHits=false; 
      }
      
    }
    if (validEcalRecHits) ecalRecHitCollection = *(ecalRecHitHandle.product());




    //if ((*iPho).isEBEEGap()) continue;  //cut out gap photons


    //filling histograms to make isolation efficiencies
    if (isLoosePhoton){
	h_phoEta_Loose_->Fill( (*iPho).eta() );
	h_phoEt_Loose_->Fill( (*iPho).et() );
    }
    if (isTightPhoton){
	h_phoEta_Tight_->Fill( (*iPho).eta() );
	h_phoEt_Tight_->Fill( (*iPho).et() );
    }



    for (int cut=0; cut !=numberOfSteps_; ++cut) {  //loop over different transverse energy cuts
      double Et =  (*iPho).et();
      bool passesCuts = false;


      if ( useBinning_ && Et > cut*cutStep_ && ( (Et < (cut+1)*cutStep_)  | (cut == numberOfSteps_-1) ) ){
	passesCuts = true;
      }
      else if ( !useBinning_ && Et > cut*cutStep_ ){
	passesCuts = true;
      }

      if (passesCuts){

	//filling isolation variable histograms

	h_nTrackIsolSolidVsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).nTrkSolidConeDR04());
	h_trackPtSumSolidVsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).trkSumPtSolidConeDR04());
	h_nTrackIsolHollowVsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).nTrkHollowConeDR04());
	h_trackPtSumHollowVsEta_[cut][0]->Fill((*iPho).eta(), (*iPho).trkSumPtHollowConeDR04());
	
	h_nTrackIsolSolidVsEt_[cut][0]->Fill((*iPho).et(),(*iPho).nTrkSolidConeDR04());
	h_trackPtSumSolidVsEt_[cut][0]->Fill((*iPho).et(),(*iPho).trkSumPtSolidConeDR04());
	h_nTrackIsolHollowVsEt_[cut][0]->Fill((*iPho).et(),(*iPho).nTrkHollowConeDR04());
	h_trackPtSumHollowVsEt_[cut][0]->Fill((*iPho).et(), (*iPho).trkSumPtHollowConeDR04());
	


	fill2DHistoVector(h_nTrackIsolSolid_,(*iPho).nTrkSolidConeDR04(),cut,type);
	fill2DHistoVector(h_trackPtSumSolid_,(*iPho).trkSumPtSolidConeDR04(),cut,type);
	fill2DHistoVector(h_nTrackIsolHollow_,(*iPho).nTrkHollowConeDR04(),cut,type);
	fill2DHistoVector(h_trackPtSumHollow_,(*iPho).trkSumPtSolidConeDR04(),cut,type);


  
	fill2DHistoVector(h_ecalSumVsEta_,(*iPho).eta(), (*iPho).ecalRecHitSumEtConeDR04(),cut,type);
	fill2DHistoVector(h_hcalSumVsEta_,(*iPho).eta(), (*iPho).hcalTowerSumEtConeDR04(),cut,type);
 	fill2DHistoVector(h_ecalSumVsEt_,(*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04(),cut,type);
 	fill2DHistoVector(h_hcalSumVsEt_,(*iPho).et(), (*iPho).hcalTowerSumEtConeDR04(),cut,type);
	fill2DHistoVector(h_ecalSum_,(*iPho).ecalRecHitSumEtConeDR04(),cut,type);
	fill2DHistoVector(h_hcalSum_,(*iPho).hcalTowerSumEtConeDR04(),cut,type);



	fill2DHistoVector(p_nTrackIsolSolidVsEta_,(*iPho).eta(),(*iPho).nTrkSolidConeDR04(),cut,type);
	fill2DHistoVector(p_trackPtSumSolidVsEta_,(*iPho).eta(),(*iPho).trkSumPtSolidConeDR04(),cut,type);
	fill2DHistoVector(p_nTrackIsolHollowVsEta_,(*iPho).eta(),(*iPho).nTrkHollowConeDR04(),cut,type);
	fill2DHistoVector(p_trackPtSumHollowVsEta_,(*iPho).eta(), (*iPho).trkSumPtHollowConeDR04(),cut,type);

	fill2DHistoVector(p_nTrackIsolSolidVsEt_,(*iPho).et(),(*iPho).nTrkSolidConeDR04(),cut,type);
	fill2DHistoVector(p_trackPtSumSolidVsEt_,(*iPho).et(),(*iPho).trkSumPtSolidConeDR04(),cut,type);
	fill2DHistoVector(p_nTrackIsolHollowVsEt_,(*iPho).et(),(*iPho).nTrkHollowConeDR04(),cut,type);
	fill2DHistoVector(p_trackPtSumHollowVsEt_,(*iPho).et(), (*iPho).trkSumPtHollowConeDR04(),cut,type);

    
	fill2DHistoVector(p_ecalSumVsEta_,(*iPho).eta(), (*iPho).ecalRecHitSumEtConeDR04(),cut,type);
	fill2DHistoVector(p_hcalSumVsEta_,(*iPho).eta(), (*iPho).hcalTowerSumEtConeDR04(),cut,type);
	
	
	p_ecalSumVsEt_[cut][0][0]->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
	p_ecalSumVsEt_[cut][0][part]->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
	p_ecalSumVsEt_[cut][type][0]->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());
	p_ecalSumVsEt_[cut][type][part]->Fill((*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04());

	p_hcalSumVsEt_[cut][0][0]->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
	p_hcalSumVsEt_[cut][0][part]->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
	p_hcalSumVsEt_[cut][type][0]->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());
	p_hcalSumVsEt_[cut][type][part]->Fill((*iPho).et(), (*iPho).hcalTowerSumEtConeDR04());



 
	fill3DHistoVector(h_hOverE_,(*iPho).hadronicOverEm(),cut,type,part);
	fill3DHistoVector(h_h1OverE_,(*iPho).hadronicDepth1OverEm(),cut,type,part);
	fill3DHistoVector(h_h2OverE_,(*iPho).hadronicDepth2OverEm(),cut,type,part);
	fill3DHistoVector(h_phoSigmaIetaIeta_,(*iPho).sigmaIetaIeta(),cut,type,part);


	//filling photon histograms
	
	nPho[cut][0][0]++;
	nPho[cut][0][part]++;
	nPho[cut][type][0]++;
	nPho[cut][type][part]++;


	fill3DHistoVector(h_phoE_,(*iPho).energy(),cut,type,part);
	fill3DHistoVector(h_phoEt_,(*iPho).et(),cut,type,part);
	fill3DHistoVector(h_r9_,(*iPho).r9(),cut,type,part);

	
	fill2DHistoVector(h_phoEta_,(*iPho).eta(),cut,type);	
	fill2DHistoVector(h_scEta_,(*iPho).superCluster()->eta(),cut,type);

	fill2DHistoVector(h_phoPhi_,(*iPho).phi(),cut,type);
	fill2DHistoVector(h_scPhi_,(*iPho).superCluster()->phi(),cut,type);


	fill2DHistoVector(h_r9VsEt_,(*iPho).et(),(*iPho).r9(),cut,type);
	fill2DHistoVector(h_r9VsEta_,(*iPho).eta(),(*iPho).r9(),cut,type);
	



	h_e1x5VsEt_[cut][0]->Fill((*iPho).et(),(*iPho).e1x5());
	h_e1x5VsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).e1x5());
	
	h_e2x5VsEt_[cut][0]->Fill((*iPho).et(),(*iPho).e2x5());
	h_e2x5VsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).e2x5());


	h_maxEXtalOver3x3VsEt_[cut][0]->Fill((*iPho).et(),  (*iPho).maxEnergyXtal()/(*iPho).e3x3() );
	h_maxEXtalOver3x3VsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3() );
	

	
	h_r1x5VsEt_[cut][0]->Fill((*iPho).et(),(*iPho).r1x5());
	h_r1x5VsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).r1x5());
	
	h_r2x5VsEt_[cut][0]->Fill((*iPho).et(),(*iPho).r2x5());
	h_r2x5VsEta_[cut][0]->Fill((*iPho).eta(),(*iPho).r2x5());
	


	fill2DHistoVector(h_sigmaIetaIetaVsEta_,(*iPho).eta(),(*iPho).sigmaIetaIeta(),cut,type);



	fill2DHistoVector(p_r9VsEt_,(*iPho).et(),(*iPho).r9(),cut,type);
	fill2DHistoVector(p_r9VsEta_,(*iPho).eta(),(*iPho).r9(),cut,type);
	
	fill2DHistoVector(p_e1x5VsEt_,(*iPho).et(),(*iPho).e1x5(),cut,type);
	fill2DHistoVector(p_e1x5VsEta_,(*iPho).eta(),(*iPho).e1x5(),cut,type);

	fill2DHistoVector(p_e2x5VsEt_,(*iPho).et(),(*iPho).e2x5(),cut,type);
	fill2DHistoVector(p_e2x5VsEta_,(*iPho).eta(),(*iPho).e2x5(),cut,type);

	fill2DHistoVector(p_maxEXtalOver3x3VsEt_,(*iPho).et(),(*iPho).maxEnergyXtal()/(*iPho).e3x3() ,cut,type);
	fill2DHistoVector(p_maxEXtalOver3x3VsEta_,(*iPho).eta(),(*iPho).maxEnergyXtal()/(*iPho).e3x3() ,cut,type);

	fill2DHistoVector(p_r1x5VsEt_,(*iPho).et(),(*iPho).r1x5(),cut,type);
	fill2DHistoVector(p_r1x5VsEta_,(*iPho).eta(),(*iPho).r1x5(),cut,type);

	fill2DHistoVector(p_r2x5VsEt_,(*iPho).et(),(*iPho).r2x5(),cut,type);
	fill2DHistoVector(p_r2x5VsEta_,(*iPho).eta(),(*iPho).r2x5(),cut,type);

	fill2DHistoVector(p_sigmaIetaIetaVsEta_,(*iPho).eta(),(*iPho).sigmaIetaIeta(),cut,type);

	fill2DHistoVector(p_hOverEVsEta_,(*iPho).eta(),(*iPho).hadronicOverEm(),cut,type);
	fill2DHistoVector(p_hOverEVsEt_,(*iPho).et(),(*iPho).hadronicOverEm(),cut,type);







 	bool atLeastOneDeadChannel=false;
 	for(reco::CaloCluster_iterator bcIt = (*iPho).superCluster()->clustersBegin();bcIt != (*iPho).superCluster()->clustersEnd(); ++bcIt) { //loop over basic clusters in SC
 	  for(std::vector< std::pair<DetId, float> >::const_iterator rhIt = (*bcIt)->hitsAndFractions().begin();rhIt != (*bcIt)->hitsAndFractions().end(); ++rhIt) { //loop over rec hits in basic cluster
	    
 	    for(EcalRecHitCollection::const_iterator it = ecalRecHitCollection.begin(); it !=  ecalRecHitCollection.end(); ++it) { //loop over all rec hits to find the right ones
 	      if  (rhIt->first ==  (*it).id() ) { //found the matching rechit
 		if (  (*it).recoFlag() == 9 ) { //has a bad channel
 		  atLeastOneDeadChannel=true;
 		  break;
 		}
 	      }
 	    }
 	  } 
 	}
	
 	if ( atLeastOneDeadChannel ) {
	  fill2DHistoVector(h_phoPhi_BadChannels_,(*iPho).phi(),cut,type);
	  fill2DHistoVector(h_phoEta_BadChannels_,(*iPho).eta(),cut,type);
	  fill2DHistoVector(h_phoEt_BadChannels_,(*iPho).et(),cut,type);
 	}





	// filling conversion-related histograms

	if((*iPho).hasConversionTracks()){
	  nConv[cut][0][0]++;
	  nConv[cut][0][part]++;
	  nConv[cut][type][0]++;
	  nConv[cut][type][part]++;
	}
 
	//loop over conversions


	reco::ConversionRefVector conversions = (*iPho).conversions();

	for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

	  reco::ConversionRef aConv=conversions[iConv];

	  if ( aConv->nTracks() <2 ) continue; 

	  //fill histogram for denominator of vertex reconstruction efficiency plot
	  if(cut==0) h_phoEta_Vertex_->Fill(aConv->pairMomentum().eta());

	  if ( !(aConv->conversionVertex().isValid()) ) continue;

	  fill3DHistoVector(h_phoConvE_,(*iPho).energy(),cut,type,part);
	  fill3DHistoVector(h_phoConvEt_,(*iPho).et(),cut,type,part);
	  fill3DHistoVector(h_phoConvR9_,(*iPho).r9(),cut,type,part);

	  if (cut==0 && isLoosePhoton){
	    h_convEta_Loose_->Fill( (*iPho).eta() );
	    h_convEt_Loose_->Fill( (*iPho).et() );
	  }
	  if (cut==0 && isTightPhoton){
	    h_convEta_Tight_->Fill( (*iPho).eta() );
	    h_convEt_Tight_->Fill( (*iPho).et() );
	  }

	
	  //fill2DHistoVector(h_phoConvEta_,aConv->caloCluster()[0]->eta(),cut,type);
	  //fill2DHistoVector(h_phoConvPhi_,aConv->caloCluster()[0]->phi(),cut,type);
	  
	  fill2DHistoVector(h_phoConvEta_,aConv->pairMomentum().eta(),cut,type);
	  fill2DHistoVector(h_phoConvPhi_,aConv->pairMomentum().phi(),cut,type);

	  fill2DHistoVector(h_phoConvEtaForEfficiency_,(*iPho).eta(),cut,type);
	  fill2DHistoVector(h_phoConvPhiForEfficiency_,(*iPho).phi(),cut,type);

	  
	  float chi2Prob = ChiSquaredProbability( aConv->conversionVertex().normalizedChi2(), aConv->conversionVertex().ndof() );
	  fill2DHistoVector(h_vertexChi2_,chi2Prob,cut,type);
	  
	  double convR= sqrt(aConv->conversionVertex().position().perp2());
	  double scalar = aConv->conversionVertex().position().x()*aConv->pairMomentum().x() + aConv->conversionVertex().position().y()*aConv->pairMomentum().y();
	  if ( scalar < 0 ) convR= -convR;
	  
	  
	  fill2DHistoVector(h_convVtxRvsZ_,fabs( aConv->conversionVertex().position().z() ), convR,cut,type);
	  
	  if(fabs(aConv->caloCluster()[0]->eta()) > 1.5){
	    fill2DHistoVector(h_convVtxZ_,fabs(aConv->conversionVertex().position().z()), cut,type);
	  }
	  else if(fabs(aConv->caloCluster()[0]->eta()) < 1){
	    fill2DHistoVector(h_convVtxR_,convR,cut,type);
	    
	    
	    fill2DHistoVector(h_convVtxYvsX_,aConv->conversionVertex().position().x(),  
			      aConv->conversionVertex().position().y(),cut,type);
	  }
	  
	  
	  
	  const std::vector<edm::RefToBase<reco::Track> > tracks = aConv->tracks();

	  
	  for (unsigned int i=0; i<tracks.size(); i++) {
	    fill2DHistoVector(h_tkChi2_,tracks[i]->normalizedChi2(),cut,type);
	    fill2DHistoVector(p_tkChi2VsEta_,aConv->caloCluster()[0]->eta(),tracks[i]->normalizedChi2(),cut,type);
	    fill2DHistoVector(p_dCotTracksVsEta_,aConv->caloCluster()[0]->eta(),aConv->pairCotThetaSeparation(),cut,type);	
	    fill2DHistoVector(p_nHitsVsEta_,aConv->caloCluster()[0]->eta(),float(tracks[i]->numberOfValidHits()),cut,type);
	  }

	  //calculating delta eta and delta phi of the two tracks

	  float  DPhiTracksAtVtx = -99;
	  float  dPhiTracksAtEcal= -99;
	  float  dEtaTracksAtEcal= -99;

	  float phiTk1= aConv->tracksPin()[0].phi();
	  float phiTk2= aConv->tracksPin()[1].phi();
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


	  fill3DHistoVector(h_dPhiTracksAtVtx_,DPhiTracksAtVtx,cut,type,part);
	  fill3DHistoVector(h_dPhiTracksAtEcal_,fabs(dPhiTracksAtEcal),cut,type,part);
	  fill3DHistoVector(h_dEtaTracksAtEcal_, dEtaTracksAtEcal,cut,type,part);
	  fill3DHistoVector(h_eOverPTracks_,aConv->EoverP(),cut,type,part);
	  fill3DHistoVector(h_pOverETracks_,1./aConv->EoverP(),cut,type,part);
	  fill3DHistoVector(h_dCotTracks_,aConv->pairCotThetaSeparation(),cut,type,part);


	}//end loop over conversions

      }//end loop over photons passing cuts
    }//end loop over transverse energy cuts





    //make invariant mass plots

    if (isIsolated && iPho->et()>=invMassEtCut_){

      for (reco::PhotonCollection::const_iterator iPho2=iPho+1; iPho2!=photonCollection.end(); iPho2++){
	
	edm::Ref<reco::PhotonCollection> photonref2(photonHandle, photonCounter); //note: correct to use photonCounter and not photonCounter+1 
	bool  isTightPhoton2(false), isLoosePhoton2(false);
	
	if ( !isHeavyIon_ ) {
	   isTightPhoton2 = (tightPhotonID)[photonref2];                      //since it has already been incremented earlier                                                                                                                 
	   isLoosePhoton2 = (loosePhotonID)[photonref2];
	}
	   bool isIsolated2=false;
	if ( isolationStrength_ == 0)  isIsolated2 = isLoosePhoton2;
	if ( isolationStrength_ == 1)  isIsolated2 = isTightPhoton2; 

	reco::ConversionRefVector conversions = (*iPho).conversions();
	reco::ConversionRefVector conversions2 = (*iPho2).conversions();

 	if(isIsolated2 && iPho2->et()>=invMassEtCut_){

	  math::XYZTLorentzVector p12 = iPho->p4()+iPho2->p4();
	  float gamgamMass2 = p12.Dot(p12);


	  h_invMassAllPhotons_ -> Fill(sqrt( gamgamMass2 ));


 	  if(conversions.size()!=0 && conversions[0]->nTracks() >= 2){
	    if(conversions2.size()!=0 && conversions2[0]->nTracks() >= 2) h_invMassTwoWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	    else h_invMassOneWithTracks_ -> Fill(sqrt( gamgamMass2 ));
 	  }
	  else if(conversions2.size()!=0 && conversions2[0]->nTracks() >= 2) h_invMassOneWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	  else h_invMassZeroWithTracks_ -> Fill(sqrt( gamgamMass2 ));
 	}
	
      }

    }
    
    
    
  }/// End loop over Reco photons
    

  //filling number of photons/conversions per event histograms
  for (int cut=0; cut !=numberOfSteps_; ++cut) {
    for(int type=0;type!=3;++type){
      for(int part=0;part!=3;++part){
	h_nPho_[cut][type][part]-> Fill (float(nPho[cut][type][part]));
	h_nConv_[cut][type][part]-> Fill (float(nConv[cut][type][part]));
      }
    }
  }

}




void PhotonAnalyzer::endJob()
{
  //dbe_->showDirStructure();

  if(standAlone_){
    dbe_->save(outputFileName_);
  }


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




void  PhotonAnalyzer::fill2DHistoVector(std::vector<std::vector<MonitorElement*> >& histoVector,double x, double y, int cut, int type){
  
  histoVector[cut][0]->Fill(x,y);
  histoVector[cut][type]->Fill(x,y);

}

void  PhotonAnalyzer::fill2DHistoVector(std::vector<std::vector<MonitorElement*> >& histoVector, double x, int cut, int type){

  histoVector[cut][0]->Fill(x);
  histoVector[cut][type]->Fill(x);

}

void  PhotonAnalyzer::fill3DHistoVector(std::vector<std::vector<std::vector<MonitorElement*> > >& histoVector,double x, int cut, int type, int part){
  
  histoVector[cut][0][0]->Fill(x);
  histoVector[cut][0][part]->Fill(x);
  histoVector[cut][type][0]->Fill(x);
  histoVector[cut][type][part]->Fill(x);

}





