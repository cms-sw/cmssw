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
 **  $Date: 2009/06/09 12:27:45 $ 
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

    triggerEvent_       = pset.getParameter<edm::InputTag>("triggerEvent");

    minPhoEtCut_        = pset.getParameter<double>("minPhoEtCut");   
    invMassEtCut_        = pset.getParameter<double>("invMassEtCut");  

    cutStep_            = pset.getParameter<double>("cutStep");
    numberOfSteps_      = pset.getParameter<int>("numberOfSteps");

    useBinning_         = pset.getParameter<bool>("useBinning");
    useTriggerFiltering_= pset.getParameter<bool>("useTriggerFiltering");


    isolationStrength_  = pset.getParameter<int>("isolationStrength");


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
  double barrelEtaMin = parameters_.getParameter<double>("barrelEtaMin");
  double barrelEtaMax = parameters_.getParameter<double>("barrelEtaMax");
  int barrelEtaBin = parameters_.getParameter<int>("barrelEtaBin");

 
  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");
  int    barrelPhiBin = parameters_.getParameter<int>("barrelPhiBin");

  double r9Min = parameters_.getParameter<double>("r9Min"); 
  double r9Max = parameters_.getParameter<double>("r9Max"); 
  int r9Bin = parameters_.getParameter<int>("r9Bin");

  double hOverEMin = parameters_.getParameter<double>("hOverEMin"); 
  double hOverEMax = parameters_.getParameter<double>("hOverEMax"); 
  int hOverEBin = parameters_.getParameter<int>("hOverEBin");

  double xyMin = parameters_.getParameter<double>("xyMin"); 
  double xyMax = parameters_.getParameter<double>("xyMax"); 
  int xyBin = parameters_.getParameter<int>("xyBin");

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

  double dRMin = parameters_.getParameter<double>("dRMin"); 
  double dRMax = parameters_.getParameter<double>("dRMax"); 
  int dRBin = parameters_.getParameter<int>("dRBin");

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
    
    h_invMassTwoWithTracks_= dbe_->book1D("invMassTwoWithTracks"," Two photon invariant mass: Both have tracks ",100,0,200);
    h_invMassOneWithTracks_= dbe_->book1D("invMassOneWithTracks"," Two photon invariant mass: Only one has tracks ",100,0,200);
    h_invMassZeroWithTracks_= dbe_->book1D("invMassZeroWithTracks"," Two photon invariant mass: Neither have tracks ",100,0,200);




    //Efficiency histograms

    currentFolder_.str("");
    currentFolder_ << "Egamma/PhotonAnalyzer/Efficiencies";
    dbe_->setCurrentFolder(currentFolder_.str());

    h_phoEta_Loose_ = dbe_->book1D("phoEtaLoose"," Loose Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEta_Tight_ = dbe_->book1D("phoEtaTight"," Tight Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEta_HLT_ = dbe_->book1D("phoEtaHLT"," Unfiltered Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoEt_Loose_ = dbe_->book1D("phoEtLoose"," Loose Photon Transverse Energy ", etBin,etMin, etMax);
    h_phoEt_Tight_ = dbe_->book1D("phoEtTight"," Tight Photon Transverse Energy ", etBin,etMin, etMax);
    h_phoEt_HLT_ = dbe_->book1D("phoEtHLT"," Unfiltered Photon Transverse Energy ", etBin,etMin, etMax);

    h_phoEta_Vertex_ = dbe_->book1D("phoEtaVertex"," Converted Photon with valid vertex Eta ",etaBin,etaMin, etaMax) ;

    //Triggers passed
    
    h_filters_ = dbe_->book1D("Filters","Filters Passed;;Fraction of Photons Passing",11,0,11);

    h_deltaR_ = dbe_->book1D("DeltaR","Minimum #deltaR between Photon and nearest TriggerObject;#deltaR",dRBin,dRMin,dRMax);
    h_failedPhoEta_ = dbe_->book1D("FailedPhoEta","#eta of reconstructed photons failing all HLT photon triggers;#eta",etaBin,etaMin, etaMax);
    h_failedPhoEt_ = dbe_->book1D("FailedPhoEt","Et of reconstructed photons failing all HLT photon triggers;Et (GeV)",etBin,etMin, etMax);


    for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
      
      // Photon histograms

      for(uint type=0;type!=types.size();++type){ //looping over isolation type
	
	currentFolder_.str("");
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
	dbe_->setCurrentFolder(currentFolder_.str());

	for(uint part=0;part!=parts.size();++part){ //looping over different parts of the ecal

	  h_phoE_part_.push_back(dbe_->book1D("phoE"+parts[part],types[type]+" Photon Energy: "+parts[part]+";E (GeV)", eBin,eMin, eMax));
	  h_phoEt_part_.push_back(dbe_->book1D("phoEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part]+";Et (GeV)", etBin,etMin, etMax));
	  h_r9_part_.push_back(dbe_->book1D("r9"+parts[part],types[type]+" Photon r9: "+parts[part]+";R9",r9Bin,r9Min, r9Max));
	  h_hOverE_part_.push_back(dbe_->book1D("hOverE"+parts[part],types[type]+" Photon H/E: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_h1OverE_part_.push_back(dbe_->book1D("h1OverE"+parts[part],types[type]+" Photon H/E for Depth 1: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_h2OverE_part_.push_back(dbe_->book1D("h2OverE"+parts[part],types[type]+" Photon H/E for Depth 2: "+parts[part]+";H/E",hOverEBin,hOverEMin,hOverEMax));
	  h_nPho_part_.push_back(dbe_->book1D("nPho"+parts[part],"Number of "+types[type]+" Photons per Event: "+parts[part]+";# #gamma", numberBin,numberMin,numberMax));

	}//end loop over different parts of the ecal

	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionAllEcal","Distribution of "+types[type]+" Photons in Eta/Phi: AllEcal;#phi;#eta",phiBin,phiMin,phiMax,etaBin,etaMin,etaMax));
	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionBarrel","Distribution of "+types[type]+" Photons in Eta/Phi: Barrel;#phi;#eta",barrelPhiBin,phiMin,phiMax,barrelEtaBin,barrelEtaMin,barrelEtaMax));
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
	h_scEta_isol_.push_back(dbe_->book1D("scEta",types[type]+" SuperCluster Eta;#eta ",etaBin,etaMin, etaMax)) ;
	h_scPhi_isol_.push_back(dbe_->book1D("scPhi",types[type]+" SuperCluster Phi;#phi ",phiBin,phiMin,phiMax)) ;
	h_r9VsEt_isol_.push_back(dbe_->book2D("r9VsEt2D",types[type]+" Photon r9 vs. Transverse Energy;Et (GeV);R9",etBin,etMin,etMax,r9Bin,r9Min,r9Max));
	p_r9VsEt_isol_.push_back(dbe_->book1D("r9VsEt",types[type]+" Photon r9 vs. Transverse Energy;Et (GeV);R9",etBin,etMin,etMax));
	h_r9VsEta_isol_.push_back(dbe_->book2D("r9VsEta2D",types[type]+" Photon r9 vs. #eta;#eta;R9",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max));
	p_r9VsEta_isol_.push_back(dbe_->book1D("r9VsEta",types[type]+" Photon r9 vs. #eta;#eta;R9",etaBin,etaMin,etaMax));


	h_e1x5VsEt_isol_.push_back(dbe_->book2D("e1x5VsEt2D",types[type]+" Photon e1x5 vs. Transverse Energy;Et (GeV);E1X5",etBin,etMin,etMax,etBin,etMin,etMax));
	p_e1x5VsEt_isol_.push_back(dbe_->book1D("e1x5VsEt",types[type]+" Photon e1x5 vs. Transverse Energy;Et (GeV);E1X5",etBin,etMin,etMax));
	h_e1x5VsEta_isol_.push_back(dbe_->book2D("e1x5VsEta2D",types[type]+" Photon e1x5 vs. #eta;#eta;E1X5",etaBin,etaMin,etaMax,etBin,etMin,etMax));
	p_e1x5VsEta_isol_.push_back(dbe_->book1D("e1x5VsEta",types[type]+" Photon e1x5 vs. #eta;#eta;E1X5",etaBin,etaMin,etaMax));

	h_e2x5VsEt_isol_.push_back(dbe_->book2D("e2x5VsEt2D",types[type]+" Photon e2x5 vs. Transverse Energy;Et (GeV);E2X5",etBin,etMin,etMax,etBin,etMin,etMax));
	p_e2x5VsEt_isol_.push_back(dbe_->book1D("e2x5VsEt",types[type]+" Photon e2x5 vs. Transverse Energy;Et (GeV);E2X5",etBin,etMin,etMax));
	h_e2x5VsEta_isol_.push_back(dbe_->book2D("e2x5VsEta2D",types[type]+" Photon e2x5 vs. #eta;#eta;E2X5",etaBin,etaMin,etaMax,etBin,etMin,etMax));
	p_e2x5VsEta_isol_.push_back(dbe_->book1D("e2x5VsEta",types[type]+" Photon e2x5 vs. #eta;#eta;E2X5",etaBin,etaMin,etaMax));

	h_r1x5VsEt_isol_.push_back(dbe_->book2D("r1x5VsEt2D",types[type]+" Photon r1x5 vs. Transverse Energy;Et (GeV);R1X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max));
	p_r1x5VsEt_isol_.push_back(dbe_->book1D("r1x5VsEt",types[type]+" Photon r1x5 vs. Transverse Energy;Et (GeV);R1X5",etBin,etMin,etMax));
	h_r1x5VsEta_isol_.push_back(dbe_->book2D("r1x5VsEta2D",types[type]+" Photon r1x5 vs. #eta;#eta;R1X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max));
	p_r1x5VsEta_isol_.push_back(dbe_->book1D("r1x5VsEta",types[type]+" Photon r1x5 vs. #eta;#eta;R1X5",etaBin,etaMin,etaMax));

	h_r2x5VsEt_isol_.push_back(dbe_->book2D("r2x5VsEt2D",types[type]+" Photon r2x5 vs. Transverse Energy;Et (GeV);R2X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max));
	p_r2x5VsEt_isol_.push_back(dbe_->book1D("r2x5VsEt",types[type]+" Photon r2x5 vs. Transverse Energy;Et (GeV);R2X5",etBin,etMin,etMax));
	h_r2x5VsEta_isol_.push_back(dbe_->book2D("r2x5VsEta2D",types[type]+" Photon r2x5 vs. #eta;#eta;R2X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max));
	p_r2x5VsEta_isol_.push_back(dbe_->book1D("r2x5VsEta",types[type]+" Photon r2x5 vs. #eta;#eta;R2X5",etaBin,etaMin,etaMax));





	h_phoSigmaIetaIeta_isol_.push_back(dbe_->book1D("phoSigmaIetaIeta",types[type]+" Photon #sigmai#etai#eta;#sigmai#etai#eta ",sigmaIetaBin,sigmaIetaMin,sigmaIetaMax)) ;
	h_sigmaIetaIetaVsEta_isol_.push_back(dbe_->book2D("sigmaIetaIetaVsEta2D",types[type]+" Photon #sigmai#etai#eta vs. #eta;#eta;#sigmai#etai#eta",etaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax));
	p_sigmaIetaIetaVsEta_isol_.push_back(dbe_->book1D("sigmaIetaIetaVsEta",types[type]+" Photon #sigmai#etai#eta vs. #eta;#eta;#sigmai#etai#eta",etaBin,etaMin,etaMax));

	h_phoSigmaEtaEta_isol_.push_back(dbe_->book1D("phoSigmaEtaEta",types[type]+" Photon #sigma#eta#eta;#sigma#eta#eta ",sigmaIetaBin,sigmaIetaMin,sigmaIetaMax)) ;
	h_sigmaEtaEtaVsEta_isol_.push_back(dbe_->book2D("sigmaEtaEtaVsEta2D",types[type]+" Photon #sigma#eta#eta vs. #eta;#eta;#sigma#eta#eta",etaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax));
	p_sigmaEtaEtaVsEta_isol_.push_back(dbe_->book1D("sigmaEtaEtaVsEta",types[type]+" Photon #sigma#eta#eta vs. #eta;#eta;#sigma#eta#eta",etaBin,etaMin,etaMax));

	// Isolation Variable infos

	h_nTrackIsolSolidVsEta_isol_.push_back(dbe_->book2D("nIsoTracksSolidVsEta2D","Avg Number Of Tracks in the Solid Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax));
	h_trackPtSumSolidVsEta_isol_.push_back(dbe_->book2D("isoPtSumSolidVsEta2D","Avg Tracks Pt Sum in the Solid Iso Cone",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax));
	h_nTrackIsolHollowVsEta_isol_.push_back(dbe_->book2D("nIsoTracksHollowVsEta2D","Avg Number Of Tracks in the Hollow Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax));
	h_trackPtSumHollowVsEta_isol_.push_back(dbe_->book2D("isoPtSumHollowVsEta2D","Avg Tracks Pt Sum in the Hollow Iso Cone",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax));
	h_ecalSumVsEta_isol_.push_back(dbe_->book2D("ecalSumVsEta2D","Avg Ecal Sum in the Iso Cone",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax));
	h_hcalSumVsEta_isol_.push_back(dbe_->book2D("hcalSumVsEta2D","Avg Hcal Sum in the Iso Cone",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax));
	p_nTrackIsolSolidVsEta_isol_.push_back(dbe_->book1D("nIsoTracksSolidVsEta","Avg Number Of Tracks in the Solid Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax));
	p_trackPtSumSolidVsEta_isol_.push_back(dbe_->book1D("isoPtSumSolidVsEta","Avg Tracks Pt Sum in the Solid Iso Cone vs.  #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax));
	p_nTrackIsolHollowVsEta_isol_.push_back(dbe_->book1D("nIsoTracksHollowVsEta","Avg Number Of Tracks in the Hollow Iso Cone vs.  #eta;#eta;# tracks",etaBin,etaMin, etaMax));
	p_trackPtSumHollowVsEta_isol_.push_back(dbe_->book1D("isoPtSumHollowVsEta","Avg Tracks Pt Sum in the Hollow Iso Cone vs.  #eta;#eta;Pt (GeV)",etaBin,etaMin, etaMax));
	p_ecalSumVsEta_isol_.push_back(dbe_->book1D("ecalSumVsEta","Avg Ecal Sum in the Iso Cone vs.  #eta;#eta;E (GeV)",etaBin,etaMin, etaMax));
	p_hcalSumVsEta_isol_.push_back(dbe_->book1D("hcalSumVsEta","Avg Hcal Sum in the Iso Cone vs.  #eta;#eta;E (GeV)",etaBin,etaMin, etaMax));

    // Ted adding histos
    

	h_nTrackIsolSolidVsEt_isol_.push_back(dbe_->book2D("nIsoTracksSolidVsEt2D","Avg Number Of Tracks in the Solid Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin, etMax,numberBin,numberMin,numberMax));
	h_trackPtSumSolidVsEt_isol_.push_back(dbe_->book2D("isoPtSumSolidVsEt2D","Avg Tracks Pt Sum in the Solid Iso Cone",etBin,etMin, etMax,sumBin,sumMin,sumMax));
	h_nTrackIsolHollowVsEt_isol_.push_back(dbe_->book2D("nIsoTracksHollowVsEt2D","Avg Number Of Tracks in the Hollow Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin, etMax,numberBin,numberMin,numberMax));
	h_trackPtSumHollowVsEt_isol_.push_back(dbe_->book2D("isoPtSumHollowVsEt2D","Avg Tracks Pt Sum in the Hollow Iso Cone",etBin,etMin, etMax,sumBin,sumMin,sumMax));
	h_ecalSumVsEt_isol_.push_back(dbe_->book2D("ecalSumVsEt2D","Avg Ecal Sum in the Iso Cone",etBin,etMin, etMax,sumBin,sumMin,sumMax));
	h_hcalSumVsEt_isol_.push_back(dbe_->book2D("hcalSumVsEt2D","Avg Hcal Sum in the Iso Cone",etBin,etMin, etMax,sumBin,sumMin,sumMax));
	p_nTrackIsolSolidVsEt_isol_.push_back(dbe_->book1D("nIsoTracksSolidVsEt","Avg Number Of Tracks in the Solid Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin, etMax));
	p_trackPtSumSolidVsEt_isol_.push_back(dbe_->book1D("isoPtSumSolidVsEt","Avg Tracks Pt Sum in the Solid Iso Cone vs.  E_{T};E_{T};Pt (GeV)",etBin,etMin, etMax));
	p_nTrackIsolHollowVsEt_isol_.push_back(dbe_->book1D("nIsoTracksHollowVsEt","Avg Number Of Tracks in the Hollow Iso Cone vs.  E_{T};E_{T};# tracks",etBin,etMin, etMax));
	p_trackPtSumHollowVsEt_isol_.push_back(dbe_->book1D("isoPtSumHollowVsEt","Avg Tracks Pt Sum in the Hollow Iso Cone vs.  E_{T};E_{T};Pt (GeV)",etBin,etMin, etMax));
	p_ecalSumVsEt_isol_.push_back(dbe_->book1D("ecalSumVsEt","Avg Ecal Sum in the Iso Cone vs.  E_{T};E_{T};E (GeV)",etBin,etMin, etMax));
	p_hcalSumVsEt_isol_.push_back(dbe_->book1D("hcalSumVsEt","Avg Hcal Sum in the Iso Cone vs.  E_{T};E_{T};E (GeV)",etBin,etMin, etMax));


    // end Ted adding histos
    
	h_nTrackIsolSolid_isol_.push_back(dbe_->book1D("nIsoTracksSolid","Avg Number Of Tracks in the Solid Iso Cone;# tracks",numberBin,numberMin,numberMax));
	h_trackPtSumSolid_isol_.push_back(dbe_->book1D("isoPtSumSolid","Avg Tracks Pt Sum in the Solid Iso Cone;Pt (GeV)",sumBin,sumMin,sumMax));
	h_nTrackIsolHollow_isol_.push_back(dbe_->book1D("nIsoTracksHollow","Avg Number Of Tracks in the Hollow Iso Cone;# tracks",numberBin,numberMin,numberMax));
	h_trackPtSumHollow_isol_.push_back(dbe_->book1D("isoPtSumHollow","Avg Tracks Pt Sum in the Hollow Iso Cone;Pt (GeV)",sumBin,sumMin,sumMax));
	h_ecalSum_isol_.push_back(dbe_->book1D("ecalSum","Avg Ecal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax));
	h_hcalSum_isol_.push_back(dbe_->book1D("hcalSum","Avg Hcal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax));


      }//end loop over isolation type

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

      h_scEta_.push_back(h_scEta_isol_);
      h_scEta_isol_.clear();
      h_scPhi_.push_back(h_scPhi_isol_);
      h_scPhi_isol_.clear();

      h_r9VsEt_.push_back(h_r9VsEt_isol_);
      h_r9VsEt_isol_.clear();
      p_r9VsEt_.push_back(p_r9VsEt_isol_);
      p_r9VsEt_isol_.clear();
      h_r9VsEta_.push_back(h_r9VsEta_isol_);
      h_r9VsEta_isol_.clear();
      p_r9VsEta_.push_back(p_r9VsEta_isol_);
      p_r9VsEta_isol_.clear();

      h_e1x5VsEt_.push_back(h_e1x5VsEt_isol_);
      h_e1x5VsEt_isol_.clear();
      p_e1x5VsEt_.push_back(p_e1x5VsEt_isol_);
      p_e1x5VsEt_isol_.clear();
      h_e1x5VsEta_.push_back(h_e1x5VsEta_isol_);
      h_e1x5VsEta_isol_.clear();
      p_e1x5VsEta_.push_back(p_e1x5VsEta_isol_);
      p_e1x5VsEta_isol_.clear();

      h_e2x5VsEt_.push_back(h_e2x5VsEt_isol_);
      h_e2x5VsEt_isol_.clear();
      p_e2x5VsEt_.push_back(p_e2x5VsEt_isol_);
      p_e2x5VsEt_isol_.clear();
      h_e2x5VsEta_.push_back(h_e2x5VsEta_isol_);
      h_e2x5VsEta_isol_.clear();
      p_e2x5VsEta_.push_back(p_e2x5VsEta_isol_);
      p_e2x5VsEta_isol_.clear();

      h_r1x5VsEt_.push_back(h_r1x5VsEt_isol_);
      h_r1x5VsEt_isol_.clear();
      p_r1x5VsEt_.push_back(p_r1x5VsEt_isol_);
      p_r1x5VsEt_isol_.clear();
      h_r1x5VsEta_.push_back(h_r1x5VsEta_isol_);
      h_r1x5VsEta_isol_.clear();
      p_r1x5VsEta_.push_back(p_r1x5VsEta_isol_);
      p_r1x5VsEta_isol_.clear();

      h_r2x5VsEt_.push_back(h_r2x5VsEt_isol_);
      h_r2x5VsEt_isol_.clear();
      p_r2x5VsEt_.push_back(p_r2x5VsEt_isol_);
      p_r2x5VsEt_isol_.clear();
      h_r2x5VsEta_.push_back(h_r2x5VsEta_isol_);
      h_r2x5VsEta_isol_.clear();
      p_r2x5VsEta_.push_back(p_r2x5VsEta_isol_);
      p_r2x5VsEta_isol_.clear();

 
      h_phoSigmaIetaIeta_.push_back(h_phoSigmaIetaIeta_isol_);
      h_phoSigmaIetaIeta_isol_.clear();
  
      h_sigmaIetaIetaVsEta_.push_back(h_sigmaIetaIetaVsEta_isol_);
      h_sigmaIetaIetaVsEta_isol_.clear();
      p_sigmaIetaIetaVsEta_.push_back(p_sigmaIetaIetaVsEta_isol_);
      p_sigmaIetaIetaVsEta_isol_.clear();

      h_phoSigmaEtaEta_.push_back(h_phoSigmaEtaEta_isol_);
      h_phoSigmaEtaEta_isol_.clear();
  
      h_sigmaEtaEtaVsEta_.push_back(h_sigmaEtaEtaVsEta_isol_);
      h_sigmaEtaEtaVsEta_isol_.clear();
      p_sigmaEtaEtaVsEta_.push_back(p_sigmaEtaEtaVsEta_isol_);
      p_sigmaEtaEtaVsEta_isol_.clear();
 
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

      h_nTrackIsolSolidVsEt_.push_back(h_nTrackIsolSolidVsEt_isol_);
      h_trackPtSumSolidVsEt_.push_back(h_trackPtSumSolidVsEt_isol_);
      h_nTrackIsolHollowVsEt_.push_back(h_nTrackIsolHollowVsEt_isol_);
      h_trackPtSumHollowVsEt_.push_back(h_trackPtSumHollowVsEt_isol_);
      h_ecalSumVsEt_.push_back(h_ecalSumVsEt_isol_);
      h_hcalSumVsEt_.push_back(h_hcalSumVsEt_isol_);
      p_nTrackIsolSolidVsEt_.push_back(p_nTrackIsolSolidVsEt_isol_);
      p_trackPtSumSolidVsEt_.push_back(p_trackPtSumSolidVsEt_isol_);
      p_nTrackIsolHollowVsEt_.push_back(p_nTrackIsolHollowVsEt_isol_);
      p_trackPtSumHollowVsEt_.push_back(p_trackPtSumHollowVsEt_isol_);
      p_ecalSumVsEt_.push_back(p_ecalSumVsEt_isol_);
      p_hcalSumVsEt_.push_back(p_hcalSumVsEt_isol_);
    
      h_nTrackIsolSolidVsEt_isol_.clear();
      h_trackPtSumSolidVsEt_isol_.clear();
      h_nTrackIsolHollowVsEt_isol_.clear();
      h_trackPtSumHollowVsEt_isol_.clear();
      h_ecalSumVsEt_isol_.clear();
      h_hcalSumVsEt_isol_.clear();
      p_nTrackIsolSolidVsEt_isol_.clear();
      p_trackPtSumSolidVsEt_isol_.clear();
      p_nTrackIsolHollowVsEt_isol_.clear();
      p_trackPtSumHollowVsEt_isol_.clear();
      p_ecalSumVsEt_isol_.clear();
      p_hcalSumVsEt_isol_.clear();

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

	//Conversion fraction histograms

	h_phoConvEta_isol_.push_back(dbe_->book1D("phoConvEta",types[type]+" Converted Photon Eta;#eta ",etaBin,etaMin, etaMax)) ;
	h_phoConvPhi_isol_.push_back(dbe_->book1D("phoConvPhi",types[type]+" Converted Photon Phi;#phi ",phiBin,phiMin,phiMax)) ;

	h_convVtxRvsZ_isol_.push_back(dbe_->book2D("convVtxRvsZ",types[type]+" Photon Reco conversion vtx position;Z (cm);R (cm)",zBin,zMin,zMax,rBin,rMin,rMax));
	h_convVtxZ_isol_.push_back(dbe_->book1D("convVtxZ",types[type]+" Photon Reco conversion vtx position: #eta > 1.5;Z (cm)",zBin,zMin,zMax));
	h_convVtxR_isol_.push_back(dbe_->book1D("convVtxR",types[type]+" Photon Reco conversion vtx position: #eta < 1;R (cm)",rBin,rMin,rMax));
	h_convVtxYvsX_isol_.push_back(dbe_->book2D("convVtxYvsX",types[type]+" Photon Reco conversion vtx position: #eta < 1;X (cm);Y (cm)",xBin,xMin,xMax,yBin,yMin,yMax));




	h_nHitsVsEta_isol_.push_back(dbe_->book2D("nHitsVsEta2D",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta;#eta;# hits",etaBin,etaMin, etaMax,etaBin,0, 16));
	p_nHitsVsEta_isol_.push_back(dbe_->book1D("nHitsVsEta",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta;#eta;# hits",etaBin,etaMin, etaMax));
	h_tkChi2VsEta_isol_.push_back(dbe_->book2D("tkChi2VsEta2D",types[type]+" Photons: Tracks from conversions: #chi^{2} vs Eta;#eta;#chi^{2}",etaBin,etaMin, etaMax,100, 0., 20.0));
	p_tkChi2VsEta_isol_.push_back(dbe_->book1D("tkChi2VsEta",types[type]+" Photons: Tracks from conversions: #chi^{2} vs Eta;#eta;#chi^{2}",etaBin,etaMin, etaMax));
	h_dCotTracksVsEta_isol_.push_back(dbe_->book2D("dCotTracksVsEta2D",types[type]+" #delta cotg(#Theta) of Conversion Tracks vs Eta;#eta;#delta cotg(#Theta)",etaBin,etaMin, etaMax,dEtaTracksBin,dEtaTracksMin,dEtaTracksMax));
	p_dCotTracksVsEta_isol_.push_back(dbe_->book1D("dCotTracksVsEta",types[type]+" #delta cotg(#Theta) of Conversion Tracks  vs Eta;#eta;#delta cotg(#Theta)",etaBin,etaMin, etaMax));

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
      h_convVtxRvsZ_.push_back(h_convVtxRvsZ_isol_);
      h_convVtxRvsZ_isol_.clear();
      h_convVtxR_.push_back(h_convVtxR_isol_);
      h_convVtxR_isol_.clear();
      h_convVtxZ_.push_back(h_convVtxZ_isol_);
      h_convVtxZ_isol_.clear();
      h_convVtxYvsX_.push_back(h_convVtxYvsX_isol_);
      h_convVtxYvsX_isol_.clear();

      h_tkChi2_.push_back(h_tkChi2_isol_);
      h_tkChi2_isol_.clear();
      h_vertexChi2_.push_back(h_vertexChi2_isol_);
      h_vertexChi2_isol_.clear();

      h_nHitsVsEta_.push_back(h_nHitsVsEta_isol_);
      h_nHitsVsEta_isol_.clear(); 
      p_nHitsVsEta_.push_back(p_nHitsVsEta_isol_);
      p_nHitsVsEta_isol_.clear();
      h_tkChi2VsEta_.push_back(h_tkChi2VsEta_isol_);
      h_tkChi2VsEta_isol_.clear(); 
      p_tkChi2VsEta_.push_back(p_tkChi2VsEta_isol_);
      p_tkChi2VsEta_isol_.clear();
      h_dCotTracksVsEta_.push_back(h_dCotTracksVsEta_isol_);
      h_dCotTracksVsEta_isol_.clear(); 
      p_dCotTracksVsEta_.push_back(p_dCotTracksVsEta_isol_);
      p_dCotTracksVsEta_isol_.clear();


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
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  e.getByLabel(triggerEvent_,triggerEventHandle);
  if(!triggerEventHandle.isValid()) {
    edm::LogInfo("PhotonProducer") << "Error! Can't get the product "<<triggerEvent_.label() << endl;
    return;
  }
  const trigger::TriggerEvent *triggerEvent = triggerEventHandle.product();


  // Get the reconstructed photons
  Handle<reco::PhotonCollection> photonHandle; 
  e.getByLabel(photonProducer_, photonCollection_ , photonHandle);
  if ( !photonHandle.isValid()) return;
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
 
  // Get the PhotonId objects
  Handle<edm::ValueMap<bool> > loosePhotonFlag;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDLoose", loosePhotonFlag);
  const edm::ValueMap<bool> *loosePhotonID = loosePhotonFlag.product();
  Handle<edm::ValueMap<bool> > tightPhotonFlag;
  e.getByLabel("PhotonIDProd", "PhotonCutBasedIDTight", tightPhotonFlag);
  const edm::ValueMap<bool> *tightPhotonID = tightPhotonFlag.product();

 

  // Create array to hold #photons/event information
  int nPho[100][3][3];

  for (int cut=0; cut!=100; ++cut){
    for (int type=0; type!=3; ++type){
      for (int part=0; part!=3; ++part){
	nPho[cut][type][part] = 0;
      }
    }
  }


 
  //Prepare list of photon-related HLT filter names

  std::vector<int> Keys;

  TH1 *filters = h_filters_->getTH1();

  if(nEvt_ == 1) {
    filters->GetXaxis()->SetBinLabel(1,"hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter");
    filters->GetXaxis()->SetBinLabel(2,"hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter");
    filters->GetXaxis()->SetBinLabel(3,"hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter");
    filters->GetXaxis()->SetBinLabel(4,"hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter");
    filters->GetXaxis()->SetBinLabel(5,"hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter");
    filters->GetXaxis()->SetBinLabel(6,"hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter");
    filters->GetXaxis()->SetBinLabel(7,"hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter");
    filters->GetXaxis()->SetBinLabel(8,"hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter");
    filters->GetXaxis()->SetBinLabel(9,"hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilterESet70");
    filters->GetXaxis()->SetBinLabel(10,"hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter");
    filters->GetXaxis()->SetBinLabel(11,"hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter");
  }


  for(uint filterIndex=0;filterIndex<triggerEvent->sizeFilters();++filterIndex){  //loop over all trigger filters in event (i.e. filters passed)

    string label = triggerEvent->filterTag(filterIndex).label();

    if(label.find( "Photon" ) != std::string::npos ) {  //get photon-related filters and fill histo

      if (label=="hltL1NonIsoHLTLEITISinglePhotonEt10TrackIsolFilter")h_filters_->Fill(0);
      if (label=="hltL1NonIsoHLTLEITISinglePhotonEt20TrackIsolFilter")h_filters_->Fill(1);
      if (label=="hltL1NonIsoHLTLEITISinglePhotonEt25TrackIsolFilter")h_filters_->Fill(2);
      if (label=="hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter")h_filters_->Fill(3);      
      if (label=="hltL1NonIsoHLTNonIsoDoublePhotonEt15HcalIsolFilter")h_filters_->Fill(4);
      if (label=="hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter")h_filters_->Fill(5);
      if (label=="hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter")h_filters_->Fill(6);
      if (label=="hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter")h_filters_->Fill(7);
      if (label=="hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilterESet70")h_filters_->Fill(8);
      if (label=="hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter")h_filters_->Fill(9);
      if (label=="hltL1NonIsoHLTVLEIDoublePhotonEt15HcalIsolFilter")h_filters_->Fill(10);

      for(uint filterKeyIndex=0;filterKeyIndex<triggerEvent->filterKeys(filterIndex).size();++filterKeyIndex){  //loop over keys to objects passing this filter
	Keys.push_back(triggerEvent->filterKeys(filterIndex)[filterKeyIndex]);  //add keys to a vector for later reference	
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

      deltaR = reco::deltaR(triggerEvent->getObjects()[(*objectKey)].eta(),triggerEvent->getObjects()[(*objectKey)].phi(),(*iPho).superCluster()->eta(),(*iPho).superCluster()->phi());
      if(deltaR < deltaRMin) deltaRMin = deltaR;
      
    }
    h_deltaR_->Fill(deltaRMin);
    
    if(deltaRMin > deltaRMax) {  //photon fails delta R cut
      h_failedPhoEta_->Fill((*iPho).superCluster()->eta());
      h_failedPhoEt_->Fill((*iPho).et());
      if(useTriggerFiltering_) continue;  //throw away photons that haven't passed any photon filters
    }

    
    if ((*iPho).et()  < minPhoEtCut_) continue;
    
    nEntry_++;
       
    edm::Ref<reco::PhotonCollection> photonref(photonHandle, photonCounter);
    photonCounter++;
    bool  isLoosePhoton = (*loosePhotonID)[photonref];
    bool  isTightPhoton = (*tightPhotonID)[photonref];


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

    /////  From 30X Photons are already pre-selected at reconstruction level with a looseEM isolation
    bool isIsolated=false;
    if ( isolationStrength_ == 0)  isIsolated = isLoosePhoton;
    if ( isolationStrength_ == 1)  isIsolated = isTightPhoton; 

    int type=0;
    if ( isIsolated ) type=1;
    if ( !isIsolated ) type=2;


    //filling histograms to make isolation efficiencies
    if (isLoosePhoton){
	h_phoEta_Loose_->Fill( (*iPho).eta() );
	h_phoEt_Loose_->Fill( (*iPho).et() );
    }
    if (isTightPhoton){
	h_phoEta_Tight_->Fill( (*iPho).eta() );
	h_phoEt_Tight_->Fill( (*iPho).et() );
    }


    if ((*iPho).isEBEEGap()) continue;  //cut out gap photons



    for (int cut=0; cut !=numberOfSteps_; ++cut) {  //loop over different transverse energy cuts
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

	fill2DHistoVector(h_nTrackIsolSolidVsEta_,(*iPho).eta(),(*iPho).nTrkSolidConeDR04(),cut,type);
	fill2DHistoVector(h_trackPtSumSolidVsEta_,(*iPho).eta(),(*iPho).trkSumPtSolidConeDR04(),cut,type);
	fill2DHistoVector(h_nTrackIsolHollowVsEta_,(*iPho).eta(),(*iPho).nTrkHollowConeDR04(),cut,type);
	fill2DHistoVector(h_trackPtSumHollowVsEta_,(*iPho).eta(), (*iPho).trkSumPtHollowConeDR04(),cut,type);
	fill2DHistoVector(h_nTrackIsolSolid_,(*iPho).nTrkSolidConeDR04(),cut,type);
	fill2DHistoVector(h_trackPtSumSolid_,(*iPho).trkSumPtSolidConeDR04(),cut,type);
	fill2DHistoVector(h_nTrackIsolHollow_,(*iPho).nTrkHollowConeDR04(),cut,type);
	fill2DHistoVector(h_trackPtSumHollow_,(*iPho).trkSumPtSolidConeDR04(),cut,type);
    
	fill2DHistoVector(h_ecalSumVsEta_,(*iPho).eta(), (*iPho).ecalRecHitSumEtConeDR04(),cut,type);
	fill2DHistoVector(h_hcalSumVsEta_,(*iPho).eta(), (*iPho).hcalTowerSumEtConeDR04(),cut,type);
	fill2DHistoVector(h_ecalSum_,(*iPho).ecalRecHitSumEtConeDR04(),cut,type);
	fill2DHistoVector(h_hcalSum_,(*iPho).hcalTowerSumEtConeDR04(),cut,type);

	fill3DHistoVector(h_hOverE_,(*iPho).hadronicOverEm(),cut,type,part);
	fill3DHistoVector(h_h1OverE_,(*iPho).hadronicDepth1OverEm(),cut,type,part);
	fill3DHistoVector(h_h2OverE_,(*iPho).hadronicDepth2OverEm(),cut,type,part);


 	fill2DHistoVector(h_nTrackIsolSolidVsEt_,(*iPho).et(),(*iPho).nTrkSolidConeDR04(),cut,type);
 	fill2DHistoVector(h_trackPtSumSolidVsEt_,(*iPho).et(),(*iPho).trkSumPtSolidConeDR04(),cut,type);
 	fill2DHistoVector(h_nTrackIsolHollowVsEt_,(*iPho).et(),(*iPho).nTrkHollowConeDR04(),cut,type);
 	fill2DHistoVector(h_trackPtSumHollowVsEt_,(*iPho).et(), (*iPho).trkSumPtHollowConeDR04(),cut,type);
 	fill2DHistoVector(h_ecalSumVsEt_,(*iPho).et(), (*iPho).ecalRecHitSumEtConeDR04(),cut,type);
 	fill2DHistoVector(h_hcalSumVsEt_,(*iPho).et(), (*iPho).hcalTowerSumEtConeDR04(),cut,type);

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

	fill2DHistoVector(h_e1x5VsEt_,(*iPho).et(),(*iPho).e1x5(),cut,type);
	fill2DHistoVector(h_e1x5VsEta_,(*iPho).eta(),(*iPho).e1x5(),cut,type);

	fill2DHistoVector(h_e2x5VsEt_,(*iPho).et(),(*iPho).e2x5(),cut,type);
	fill2DHistoVector(h_e2x5VsEta_,(*iPho).eta(),(*iPho).e2x5(),cut,type);

	fill2DHistoVector(h_r1x5VsEt_,(*iPho).et(),(*iPho).r1x5(),cut,type);
	fill2DHistoVector(h_r1x5VsEta_,(*iPho).eta(),(*iPho).r1x5(),cut,type);

	fill2DHistoVector(h_r2x5VsEt_,(*iPho).et(),(*iPho).r2x5(),cut,type);
	fill2DHistoVector(h_r2x5VsEta_,(*iPho).eta(),(*iPho).r2x5(),cut,type);

	fill2DHistoVector(h_phoSigmaIetaIeta_,(*iPho).sigmaIetaIeta(),cut,type);
	fill2DHistoVector(h_sigmaIetaIetaVsEta_,(*iPho).eta(),(*iPho).sigmaIetaIeta(),cut,type);
	fill2DHistoVector(h_phoSigmaEtaEta_,(*iPho).sigmaEtaEta(),cut,type);
	fill2DHistoVector(h_sigmaEtaEtaVsEta_,(*iPho).eta(),(*iPho).sigmaEtaEta(),cut,type);

	h_phoDistribution_[cut][0][0]->Fill( (*iPho).phi(),(*iPho).eta() );
	h_phoDistribution_[cut][type][0]->Fill( (*iPho).phi(),(*iPho).eta() );
	if ( phoIsInBarrel ) {
	  h_phoDistribution_[cut][0][1]->Fill( (*iPho).phi(),(*iPho).eta() );
	  h_phoDistribution_[cut][type][1]->Fill( (*iPho).phi(),(*iPho).eta() );
	}	
	if ( phoIsInEndcapMinus ) {
	  h_phoDistribution_[cut][0][2]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	  h_phoDistribution_[cut][type][2]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	}	
	if ( phoIsInEndcapPlus ) {
	  h_phoDistribution_[cut][0][3]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	  h_phoDistribution_[cut][type][3]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	}


	// filling conversion-related histograms

	fill3DHistoVector(h_nConv_,float( (*iPho).conversions().size() ),cut,type,part);

	if((*iPho).hasConversionTracks()){
	  fill3DHistoVector(h_phoConvE_,(*iPho).energy(),cut,type,part);
	  fill3DHistoVector(h_phoConvEt_,(*iPho).et(),cut,type,part);
	  fill3DHistoVector(h_phoConvR9_,(*iPho).r9(),cut,type,part);
	}

 
	//loop over conversions

	reco::ConversionRefVector conversions = (*iPho).conversions();
	for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

	  reco::ConversionRef aConv=conversions[iConv];

	  if ( conversions[iConv]->nTracks() <2 ) continue; 

	  fill2DHistoVector(h_phoConvEta_,conversions[iConv]->caloCluster()[0]->eta(),cut,type);
	  fill2DHistoVector(h_phoConvPhi_,conversions[iConv]->caloCluster()[0]->phi(),cut,type);

	  if ( conversions[iConv]->conversionVertex().isValid() ) {

	    if(cut==0) h_phoEta_Vertex_->Fill(conversions[iConv]->caloCluster()[0]->eta());

	    float chi2Prob = ChiSquaredProbability( conversions[iConv]->conversionVertex().normalizedChi2(), conversions[iConv]->conversionVertex().ndof() );
	    fill2DHistoVector(h_vertexChi2_,chi2Prob,cut,type);


	    fill2DHistoVector(h_convVtxRvsZ_,fabs( conversions[iConv]->conversionVertex().position().z() ),  
			      sqrt( conversions[iConv]->conversionVertex().position().perp2() ),cut,type);

	    if(fabs(conversions[iConv]->caloCluster()[0]->eta()) > 1.5){
	      fill2DHistoVector(h_convVtxZ_,fabs(conversions[iConv]->conversionVertex().position().z()), cut,type);
	    }
	    else if(fabs(conversions[iConv]->caloCluster()[0]->eta()) < 1){
	      fill2DHistoVector(h_convVtxR_,sqrt( conversions[iConv]->conversionVertex().position().perp2() ),cut,type);

	      fill2DHistoVector(h_convVtxYvsX_,conversions[iConv]->conversionVertex().position().x(),  
				conversions[iConv]->conversionVertex().position().y(),cut,type);
	    }

	  }


	  std::vector<reco::TrackRef> tracks = conversions[iConv]->tracks();

	  for (unsigned int i=0; i<tracks.size(); i++) {
	    fill2DHistoVector(h_tkChi2_,tracks[i]->normalizedChi2(),cut,type);
	    fill2DHistoVector(h_tkChi2VsEta_,conversions[iConv]->caloCluster()[0]->eta(),tracks[i]->normalizedChi2(),cut,type);
	    fill2DHistoVector(h_dCotTracksVsEta_,conversions[iConv]->caloCluster()[0]->eta(),conversions[iConv]->pairCotThetaSeparation(),cut,type);	
	    fill2DHistoVector(h_nHitsVsEta_,conversions[iConv]->caloCluster()[0]->eta(),float(tracks[i]->numberOfValidHits()),cut,type);
	  }

	  //calculating delta eta and delta phi of the two tracks

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


	  fill3DHistoVector(h_dPhiTracksAtVtx_,DPhiTracksAtVtx,cut,type,part);
	  fill3DHistoVector(h_dPhiTracksAtEcal_,fabs(dPhiTracksAtEcal),cut,type,part);
	  fill3DHistoVector(h_dEtaTracksAtEcal_, dEtaTracksAtEcal,cut,type,part);
	  fill3DHistoVector(h_eOverPTracks_,conversions[iConv]->EoverP(),cut,type,part);
	  fill3DHistoVector(h_pOverETracks_,1./conversions[iConv]->EoverP(),cut,type,part);
	  fill3DHistoVector(h_dCotTracks_,conversions[iConv]->pairCotThetaSeparation(),cut,type,part);


	}//end loop over conversions

      }
    }//end loop over transverse energy cuts





    //make invariant mass plots

    if (isIsolated && iPho->et()>=invMassEtCut_){

      for (reco::PhotonCollection::const_iterator iPho2=iPho+1; iPho2!=photonCollection.end(); iPho2++){
	
	edm::Ref<reco::PhotonCollection> photonref2(photonHandle, photonCounter); //note: correct to use photonCounter and not photonCounter+1 
	bool  isTightPhoton2 = (*tightPhotonID)[photonref2];                      //since it has already been incremented earlier
	bool  isLoosePhoton2 = (*loosePhotonID)[photonref2];
	
	bool isIsolated2=false;
	if ( isolationStrength_ == 0)  isIsolated2 = isLoosePhoton2;
	if ( isolationStrength_ == 1)  isIsolated2 = isTightPhoton2; 


	if(isIsolated2 && iPho2->et()>=invMassEtCut_){

	  math::XYZTLorentzVector p12 = iPho->p4()+iPho2->p4();
	  float gamgamMass2 = p12.Dot(p12);
	  
	  if((*iPho).hasConversionTracks()){
	    if((*iPho2).hasConversionTracks()) h_invMassTwoWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	    else h_invMassOneWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	  }
	  else if((*iPho2).hasConversionTracks()) h_invMassOneWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	  else h_invMassZeroWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	}
	
      }

    }
    
    
    
  }/// End loop over Reco photons
    

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
{}



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





