#include <iostream>
//

#include "DQMOffline/EGamma/interface/PhotonAnalyzer.h"


//#define TWOPI 6.283185308
// 

/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
 **  $Date: 2008/09/30 19:50:30 $ 
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

    barrelEcalHits_     = pset.getParameter<edm::InputTag>("barrelEcalHits");
    endcapEcalHits_     = pset.getParameter<edm::InputTag>("endcapEcalHits");


    triggerResultsHLT_     = pset.getParameter<edm::InputTag>("triggerResultsHLT");
    triggerResultsFU_     = pset.getParameter<edm::InputTag>("triggerResultsFU");

    minPhoEtCut_        = pset.getParameter<double>("minPhoEtCut");   

    cutStep_            = pset.getParameter<double>("cutStep");
    numberOfSteps_      = pset.getParameter<int>("numberOfSteps");

    useBinning_         = pset.getParameter<bool>("useBinning");
    useTriggerFiltering_= pset.getParameter<bool>("useTriggerFiltering");
    standAlone_         = pset.getParameter<bool>("standAlone");

    isolationStrength_  = pset.getParameter<int>("isolationStrength");



    // parameters for Pizero finding
    seleXtalMinEnergy_    = pset.getParameter<double> ("seleXtalMinEnergy");
    clusSeedThr_          = pset.getParameter<double> ("clusSeedThr");
    clusEtaSize_          = pset.getParameter<int> ("clusEtaSize");
    clusPhiSize_          = pset.getParameter<int> ("clusPhiSize");
    ParameterLogWeighted_ = pset.getParameter<bool> ("ParameterLogWeighted");
    ParameterX0_          = pset.getParameter<double> ("ParameterX0");
    ParameterT0_barl_     = pset.getParameter<double> ("ParameterT0_barl");
    ParameterW0_          = pset.getParameter<double> ("ParameterW0");
    
    selePtGammaOne_       = pset.getParameter<double> ("selePtGammaOne");  
    selePtGammaTwo_       = pset.getParameter<double> ("selePtGammaTwo");  
    seleS4S9GammaOne_     = pset.getParameter<double> ("seleS4S9GammaOne");  
    seleS4S9GammaTwo_     = pset.getParameter<double> ("seleS4S9GammaTwo");  
    selePtPi0_            = pset.getParameter<double> ("selePtPi0");  
    selePi0Iso_           = pset.getParameter<double> ("selePi0Iso");  
    selePi0BeltDR_        = pset.getParameter<double> ("selePi0BeltDR");  
    selePi0BeltDeta_      = pset.getParameter<double> ("selePi0BeltDeta");  
    seleMinvMaxPi0_       = pset.getParameter<double> ("seleMinvMaxPi0");  
    seleMinvMinPi0_       = pset.getParameter<double> ("seleMinvMinPi0");  
  
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

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int etaBin = parameters_.getParameter<int>("etaBin");
 
  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");
 
 
  double r9Min = parameters_.getParameter<double>("r9Min"); 
  double r9Max = parameters_.getParameter<double>("r9Max"); 
  int r9Bin = parameters_.getParameter<int>("r9Bin");

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
  types.push_back("Isolated");
  types.push_back("Nonisolated");

  //booking all histograms

  if (dbe_) {  


    for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
     

      // Isolation Variable infos
     
      currentFolder_.str("");
      currentFolder_ << "Egamma/PhotonAnalyzer/IsolationVariables/Et above " << cut*cutStep_ << " GeV";
      dbe_->setCurrentFolder(currentFolder_.str());

      h_nTrackIsolSolid_.push_back(dbe_->book2D("nIsoTracksSolid2D","Avg Number Of Tracks in the Solid Iso Cone",etaBin,etaMin, etaMax,10,-0.5, 9.5));
      h_trackPtSumSolid_.push_back(dbe_->book2D("isoPtSumSolid2D","Avg Tracks Pt Sum in the Solid Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
      h_nTrackIsolHollow_.push_back(dbe_->book2D("nIsoTracksHollow2D","Avg Number Of Tracks in the Hollow Iso Cone",etaBin,etaMin, etaMax,10,-0.5, 9.5));
      h_trackPtSumHollow_.push_back(dbe_->book2D("isoPtSumHollow2D","Avg Tracks Pt Sum in the Hollow Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
      h_ecalSum_.push_back(dbe_->book2D("ecalSum2D","Avg Ecal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
      h_hcalSum_.push_back(dbe_->book2D("hcalSum2D","Avg Hcal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
      p_nTrackIsolSolid_.push_back(dbe_->book1D("nIsoTracksSolid","Avg Number Of Tracks in the Solid Iso Cone",etaBin,etaMin, etaMax));
      p_trackPtSumSolid_.push_back(dbe_->book1D("isoPtSumSolid","Avg Tracks Pt Sum in the Solid Iso Cone",etaBin,etaMin, etaMax));
      p_nTrackIsolHollow_.push_back(dbe_->book1D("nIsoTracksHollow","Avg Number Of Tracks in the Hollow Iso Cone",etaBin,etaMin, etaMax));
      p_trackPtSumHollow_.push_back(dbe_->book1D("isoPtSumHollow","Avg Tracks Pt Sum in the Hollow Iso Cone",etaBin,etaMin, etaMax));
      p_ecalSum_.push_back(dbe_->book1D("ecalSum","Avg Ecal Sum in the Iso Cone",etaBin,etaMin, etaMax));
      p_hcalSum_.push_back(dbe_->book1D("hcalSum","Avg Hcal Sum in the Iso Cone",etaBin,etaMin, etaMax));
   
      //Efficiency histograms

      p_efficiencyVsEta_.push_back(dbe_->book1D("EfficiencyVsEta","Fraction of Isolated Photons  vs. Eta",etaBin,etaMin, etaMax));
      p_efficiencyVsEt_.push_back(dbe_->book1D("EfficiencyVsEt","Fraction of Isolated Photons vs. Et",etBin,etMin, etMax));

      //Conversion fraction histograms

      p_convFractionVsEta_.push_back(dbe_->book1D("convFractionVsEta","Fraction of Converted Photons  vs. Eta",etaBin,etaMin, etaMax));
      p_convFractionVsEt_.push_back(dbe_->book1D("convFractionVsEt","Fraction of Converted Photons vs. Et",etBin,etMin, etMax));

      //Triggers passed

      currentFolder_.str("");
      currentFolder_ << "Egamma/PhotonAnalyzer/";
      dbe_->setCurrentFolder(currentFolder_.str());

      h_triggers_ = dbe_->book1D("Triggers","Triggers Passed",100,0,100);



      // Photon histograms

      for(uint type=0;type!=types.size();++type){ //looping over isolation type
	
	currentFolder_.str("");
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
	dbe_->setCurrentFolder(currentFolder_.str());

	for(uint part=0;part!=parts.size();++part){ //loop over different parts of the ecal

	  h_phoE_part_.push_back(dbe_->book1D("phoE"+parts[part],types[type]+" Photon Energy: "+parts[part], eBin,eMin, eMax));
	  h_phoEt_part_.push_back(dbe_->book1D("phoEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part], etBin,etMin, etMax));
	  h_r9_part_.push_back(dbe_->book1D("r9"+parts[part],types[type]+" Photon r9: "+parts[part],r9Bin,r9Min, r9Max));
	  h_hOverE_part_.push_back(dbe_->book1D("hOverE"+parts[part],types[type]+" Photon H/E: "+parts[part],r9Bin,r9Min, 1));
	  h_nPho_part_.push_back(dbe_->book1D("nPho"+parts[part],"Number of "+types[type]+" Photons per Event: "+parts[part], 10,-0.5,9.5));
	}

	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionAllEcal","Distribution of "+types[type]+" Photons in Eta/Phi: AllEcal",phiBin,phiMin,phiMax,etaBin,-2.5,2.5));
	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionBarrel","Distribution of "+types[type]+" Photons in Eta/Phi: Barrel",360,phiMin,phiMax,170,-1.5,1.5));
	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionEndcapMinus","Distribution of "+types[type]+" Photons in X/Y: EndcapMinus",100,-150,150,100,-150,150));
	h_phoDistribution_part_.push_back(dbe_->book2D("DistributionEndcapPlus","Distribution of "+types[type]+" Photons in X/Y: EndcapPlus",100,-150,150,100,-150,150));

	h_phoE_isol_.push_back(h_phoE_part_);
	h_phoE_part_.clear();
	h_phoEt_isol_.push_back(h_phoEt_part_);
	h_phoEt_part_.clear();
	h_r9_isol_.push_back(h_r9_part_);
	h_r9_part_.clear();
	h_hOverE_isol_.push_back(h_hOverE_part_);
	h_hOverE_part_.clear();
	h_nPho_isol_.push_back(h_nPho_part_);
	h_nPho_part_.clear();

	h_phoDistribution_isol_.push_back(h_phoDistribution_part_);
	h_phoDistribution_part_.clear();

	h_phoEta_isol_.push_back(dbe_->book1D("phoEta",types[type]+" Photon Eta ",etaBin,etaMin, etaMax)) ;
	h_phoPhi_isol_.push_back(dbe_->book1D("phoPhi",types[type]+" Photon Phi ",phiBin,phiMin,phiMax)) ;
	h_r9VsEt_isol_.push_back(dbe_->book2D("r9VsEt2D",types[type]+" Photon r9 vs. Transverse Energy",etBin,etMin,etMax,r9Bin,r9Min,r9Max));
	p_r9VsEt_isol_.push_back(dbe_->book1D("r9VsEt",types[type]+" Photon r9 vs. Transverse Energy",etBin,etMin,etMax));


      }

      h_phoE_.push_back(h_phoE_isol_);
      h_phoE_isol_.clear();
      h_phoEt_.push_back(h_phoEt_isol_);
      h_phoEt_isol_.clear();
      h_r9_.push_back(h_r9_isol_);
      h_r9_isol_.clear();
      h_hOverE_.push_back(h_hOverE_isol_);
      h_hOverE_isol_.clear();
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
      
   


      //conversion plots



      for(uint type=0;type!=types.size();++type){ //looping over isolation type

	currentFolder_.str("");	
	currentFolder_ << "Egamma/PhotonAnalyzer/" << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/Conversions";
	dbe_->setCurrentFolder(currentFolder_.str());

	for(uint part=0;part!=parts.size();++part){ //loop over different parts of the ecal

	  h_phoConvEt_part_.push_back(dbe_->book1D("phoConvEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part], etBin,etMin, etMax));

	  h_nConv_part_.push_back(dbe_->book1D("nConv"+parts[part],"Number Of Conversions per Event:  "+parts[part] ,10,-0.5, 9.5));
	  h_eOverPTracks_part_.push_back(dbe_->book1D("eOverPTracks"+parts[part],"E/P of Conversions: "+parts[part] ,100, 0., 5.));
	  
	  h_dPhiTracksAtVtx_part_.push_back(dbe_->book1D("dPhiTracksAtVtx"+parts[part], "  #delta#phi of Conversion Tracks at Vertex: "+parts[part],dPhiTracksBin,dPhiTracksMin,dPhiTracksMax));
	  h_dCotTracks_part_.push_back(dbe_->book1D("dCotTracks"+parts[part],"#delta cotg(#Theta) of Conversion Tracks: "+parts[part],dEtaTracksBin,dEtaTracksMin,dEtaTracksMax)); 

	  h_dPhiTracksAtEcal_part_.push_back(dbe_->book1D("dPhiTracksAtEcal"+parts[part], "  #delta#phi of Conversion Tracks at Ecal: "+parts[part],dPhiTracksBin,0.,dPhiTracksMax)); 
	  h_dEtaTracksAtEcal_part_.push_back(dbe_->book1D("dEtaTracksAtEcal"+parts[part], "  #delta#eta of Conversion Tracks at Ecal: "+parts[part],dEtaTracksBin,dEtaTracksMin,dEtaTracksMax)); 



	}

	h_phoConvEt_isol_.push_back(h_phoConvEt_part_);
	h_phoConvEt_part_.clear();

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




	h_phoConvEta_isol_.push_back(dbe_->book1D("phoConvEta",types[type]+" Converted Photon Eta ",etaBin,etaMin, etaMax)) ;
	h_phoConvPhi_isol_.push_back(dbe_->book1D("phoConvPhi",types[type]+" Converted Photon Phi ",phiBin,phiMin,phiMax)) ;
	h_convVtxRvsZ_isol_.push_back(dbe_->book2D("convVtxRvsZ",types[type]+" Photon Reco conversion vtx position",100, 0., 280.,200,0., 120.));
	h_convVtxRvsZLowEta_isol_.push_back(dbe_->book2D("convVtxRvsZHighEta",types[type]+" Photon Reco conversion vtx position: #eta < 1",100, 0., 280.,200,0., 120.));
	h_convVtxRvsZHighEta_isol_.push_back(dbe_->book2D("convVtxRvsZLowEta",types[type]+" Photon Reco conversion vtx position: #eta > 1",100, 0., 280.,200,0., 120.));
	h_nHitsVsEta_isol_.push_back(dbe_->book2D("nHitsVsEta2D",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta",etaBin,etaMin, etaMax,etaBin,0, 16));
	p_nHitsVsEta_isol_.push_back(dbe_->book1D("nHitsVsEta",types[type]+" Photons: Tracks from conversions: Mean Number of  Hits vs Eta",etaBin,etaMin, etaMax));	
	h_tkChi2_isol_.push_back(dbe_->book1D("tkChi2",types[type]+" Photons: Tracks from conversions: #chi^{2} of all tracks", 100, 0., 20.0));  
      }

      h_phoConvEt_.push_back(h_phoConvEt_isol_);
      h_phoConvEt_isol_.clear();
   
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


    currentFolder_.str("");
    currentFolder_ << "Egamma/PhotonAnalyzer/PiZero";
    dbe_->setCurrentFolder(currentFolder_.str());
    
    hMinvPi0EB_ = dbe_->book1D("Pi0InvmassEB","Pi0 Invariant Mass in EB",100,0.,0.5);
    hMinvPi0EB_->setAxisTitle("Inv Mass [GeV] ",1);

    hPt1Pi0EB_ = dbe_->book1D("Pt1Pi0EB","Pt 1st most energetic Pi0 photon in EB",100,0.,20.);
    hPt1Pi0EB_->setAxisTitle("1st photon Pt [GeV] ",1);
    
    hPt2Pi0EB_ = dbe_->book1D("Pt2Pi0EB","Pt 2nd most energetic Pi0 photon in EB",100,0.,20.);
    hPt2Pi0EB_->setAxisTitle("2nd photon Pt [GeV] ",1);
    
    hPtPi0EB_ = dbe_->book1D("PtPi0EB","Pi0 Pt in EB",100,0.,20.);
    hPtPi0EB_->setAxisTitle("Pi0 Pt [GeV] ",1);
    
    hIsoPi0EB_ = dbe_->book1D("IsoPi0EB","Pi0 Iso in EB",50,0.,1.);
    hIsoPi0EB_->setAxisTitle("Pi0 Iso",1);
   

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

 
  // Get EcalRecHits
  bool validEcalRecHits=true;
  Handle<EcalRecHitCollection> barrelHitHandle;
  EcalRecHitCollection barrelRecHits;
  e.getByLabel(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
    validEcalRecHits=false; 
  }
   
  Handle<EcalRecHitCollection> endcapHitHandle;
  e.getByLabel(endcapEcalHits_, endcapHitHandle);
  EcalRecHitCollection endcapRecHits;
  if (!endcapHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
    validEcalRecHits=false; 
  }


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

    if ((*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->eta())  < minPhoEtCut_) continue;
    


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



    /////  From 30X Photons are already pre-selected at reconstruction level with a looseEM isolation
    bool isIsolated=false;
    if ( isolationStrength_ == 1)  isIsolated = isLoosePhoton;
    if ( isolationStrength_ == 2)  isIsolated = isTightPhoton; 

    int type=0;
    if ( isIsolated ) type=1;
    if ( !isIsolated ) type=2;


    nEntry_++;

    float r9 = (*iPho).r9();;

    for (int cut=0; cut !=numberOfSteps_; ++cut) {
      double Et =  (*iPho).energy()/cosh( (*iPho).superCluster()->eta());

      bool passesCuts = false;

      if ( useBinning_ && Et > cut*cutStep_ && ( Et < (cut+1)*cutStep_  | cut == numberOfSteps_-1 ) ){
	passesCuts = true;
      }
      else if ( !useBinning_ && Et > cut*cutStep_ ){
	passesCuts = true;
      }

      if (passesCuts){

	//filling isolation variable histograms
	h_nTrackIsolSolid_[cut]->Fill( (*iPho).superCluster()->eta(),(*iPho).nTrkSolidConeDR04());
	h_trackPtSumSolid_[cut]->Fill((*iPho).superCluster()->eta(), (*iPho).isolationTrkSolidConeDR04());
	
	h_nTrackIsolHollow_[cut]->Fill( (*iPho).superCluster()->eta(),(*iPho).nTrkHollowConeDR04());
	h_trackPtSumHollow_[cut]->Fill((*iPho).superCluster()->eta(), (*iPho).isolationTrkHollowConeDR04());
	
	h_ecalSum_[cut]->Fill((*iPho).superCluster()->eta(), (*iPho).ecalRecHitSumConeDR04());
	h_hcalSum_[cut]->Fill((*iPho).superCluster()->eta(), (*iPho).hcalTowerSumConeDR04());


	//filling all photons histograms
	h_phoE_[cut][0][0]->Fill( (*iPho).energy() );
	h_phoEt_[cut][0][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );

	h_r9_[cut][0][0]->Fill( r9 );
	h_hOverE_[cut][0][0]->Fill( (*iPho).hadronicOverEm() );

	nPho[cut][0][0]++;
	h_nConv_[cut][0][0]->Fill(float( (*iPho).conversions().size() ));

	h_phoDistribution_[cut][0][0]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );

	h_phoEta_[cut][0]->Fill( (*iPho).superCluster()->eta() );
	h_phoPhi_[cut][0]->Fill( (*iPho).superCluster()->phi() );      

	h_r9VsEt_[cut][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()), r9 );


	// iso/noniso photons histograms
	h_phoE_[cut][type][0]->Fill( (*iPho).energy() );
	h_phoEt_[cut][type][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );

	h_r9_[cut][type][0]->Fill( r9 );
	h_hOverE_[cut][type][0]->Fill( (*iPho).hadronicOverEm() );

	nPho[cut][type][0]++;
	h_nConv_[cut][type][0]->Fill(float( (*iPho).conversions().size() ));

	h_phoDistribution_[cut][type][0]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );

	h_phoEta_[cut][type]->Fill( (*iPho).superCluster()->eta() );
	h_phoPhi_[cut][type]->Fill( (*iPho).superCluster()->phi() );      

	h_r9VsEt_[cut][type]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()), r9 );


	if((*iPho).hasConversionTracks()){
	  	h_phoConvEt_[cut][0][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	  	h_phoConvEt_[cut][type][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	}

	//filling both types of histograms for different ecal parts
	int part = 0;
	if ( phoIsInBarrel )
	  part = 1;
	if ( phoIsInEndcap )
	  part = 2;

	if((*iPho).hasConversionTracks()){
	  	h_phoConvEt_[cut][0][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	  	h_phoConvEt_[cut][type][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	}

	h_phoE_[cut][0][part]->Fill( (*iPho).energy() );
	h_phoEt_[cut][0][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );

	h_r9_[cut][0][part]->Fill( r9 );
	h_hOverE_[cut][0][part]->Fill( (*iPho).hadronicOverEm() );	

	nPho[cut][0][part]++;
	h_nConv_[cut][0][part]->Fill(float( (*iPho).conversions().size() ));

	if ( phoIsInBarrel )  h_phoDistribution_[cut][0][1]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
	if ( phoIsInEndcapMinus )  h_phoDistribution_[cut][0][2]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	if ( phoIsInEndcapPlus )  h_phoDistribution_[cut][0][3]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );


	h_phoE_[cut][type][part]->Fill( (*iPho).energy() );
	h_phoEt_[cut][type][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );

	h_r9_[cut][type][part]->Fill( r9 );
	h_hOverE_[cut][type][part]->Fill( (*iPho).hadronicOverEm() );

	nPho[cut][type][part]++;
	h_nConv_[cut][type][part]->Fill(float( (*iPho).conversions().size() ));

       	if ( phoIsInBarrel )  h_phoDistribution_[cut][type][1]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
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

  if (validEcalRecHits) makePizero(esup,  barrelHitHandle, endcapHitHandle);



}

void PhotonAnalyzer::makePizero ( const edm::EventSetup& es, const edm::Handle<EcalRecHitCollection> rhEB,  const edm::Handle<EcalRecHitCollection> rhEE ) {

  const EcalRecHitCollection *hitCollection_p = rhEB.product();
  
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);     

  edm::ESHandle<CaloTopology> theCaloTopology;
  es.get<CaloTopologyRecord>().get(theCaloTopology);

 
  const CaloSubdetectorGeometry *geometry_p;    
  const CaloSubdetectorTopology *topology_p;
  const CaloSubdetectorGeometry *geometryES_p;
  geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  // Initialize the Position Calc
  std::map<std::string,double> providedParameters;  
  providedParameters.insert(std::make_pair("LogWeighted",ParameterLogWeighted_));
  providedParameters.insert(std::make_pair("X0",ParameterX0_));
  providedParameters.insert(std::make_pair("T0_barl",ParameterT0_barl_));
  providedParameters.insert(std::make_pair("W0",ParameterW0_));
  PositionCalc posCalculator_ = PositionCalc(providedParameters);
  //
  std::map<DetId, EcalRecHit> recHitsEB_map;
  //
  std::vector<EcalRecHit> seeds;

  seeds.clear();
  //
  vector<EBDetId> usedXtals;
  usedXtals.clear();
  //
  EcalRecHitCollection::const_iterator itb;
  //
  static const int MAXCLUS = 2000;
  int nClus=0;
  vector<float> eClus;
  vector<float> etClus;
  vector<float> etaClus;
  vector<float> phiClus;
  vector<EBDetId> max_hit;
  vector< vector<EcalRecHit> > RecHitsCluster;
  vector<float> s4s9Clus;
  
  // find cluster seeds in EB 
  for(itb=rhEB->begin(); itb!=rhEB->end(); ++itb){
    EBDetId id(itb->id());
    double energy = itb->energy();
    if (energy > seleXtalMinEnergy_) {
      std::pair<DetId, EcalRecHit> map_entry(itb->id(), *itb);
      recHitsEB_map.insert(map_entry);
    }
    if (energy > clusSeedThr_) seeds.push_back(*itb);
  } // Eb rechits
  
  sort(seeds.begin(), seeds.end(), ecalRecHitLess());
  for (std::vector<EcalRecHit>::iterator itseed=seeds.begin(); itseed!=seeds.end(); itseed++) {
    EBDetId seed_id = itseed->id();
    std::vector<EBDetId>::const_iterator usedIds;
    
    bool seedAlreadyUsed=false;
    for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
      if(*usedIds==seed_id){
	seedAlreadyUsed=true;
	//cout<< " Seed with energy "<<itseed->energy()<<" was used !"<<endl;
	break; 
      }
    }
    if(seedAlreadyUsed)continue;
    topology_p = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
    std::vector<DetId> clus_v = topology_p->getWindow(seed_id,clusEtaSize_,clusPhiSize_);       
    std::vector<DetId> clus_used;

    vector<EcalRecHit> RecHitsInWindow;
    
    double simple_energy = 0; 
    
    for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
      EBDetId EBdet = *det;
      //      cout<<" det "<< EBdet<<" ieta "<<EBdet.ieta()<<" iphi "<<EBdet.iphi()<<endl;
      bool  HitAlreadyUsed=false;
      for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
	if(*usedIds==*det){
	  HitAlreadyUsed=true;
	  break;
	}
      }
      if(HitAlreadyUsed)continue;
      if (recHitsEB_map.find(*det) != recHitsEB_map.end()){
	//      cout<<" Used det "<< EBdet<<endl;
	std::map<DetId, EcalRecHit>::iterator aHit;
	aHit = recHitsEB_map.find(*det);
	usedXtals.push_back(*det);
	RecHitsInWindow.push_back(aHit->second);
	clus_used.push_back(*det);
	simple_energy = simple_energy + aHit->second.energy();
      }
    }
    
    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_p,geometry_p,geometryES_p);
    float theta_s = 2. * atan(exp(-clus_pos.eta()));
    float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
    float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
    //      float p0z_s = simple_energy * cos(theta_s);
    float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);
    
    //cout << "       Simple Clustering: E,Et,px,py,pz: "<<simple_energy<<" "<<et_s<<" "<<p0x_s<<" "<<p0y_s<<" "<<endl;

    eClus.push_back(simple_energy);
    etClus.push_back(et_s);
    etaClus.push_back(clus_pos.eta());
    phiClus.push_back(clus_pos.phi());
    max_hit.push_back(seed_id);
    RecHitsCluster.push_back(RecHitsInWindow);
    //Compute S4/S9 variable
    //We are not sure to have 9 RecHits so need to check eta and phi:
    float s4s9_[4];
    for(int i=0;i<4;i++)s4s9_[i]= itseed->energy();
    for(unsigned int j=0; j<RecHitsInWindow.size();j++){
      //cout << " Simple cluster rh, ieta, iphi : "<<((EBDetId)RecHitsInWindow[j].id()).ieta()<<" "<<((EBDetId)RecHitsInWindow[j].id()).iphi()<<endl;
      if((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()-1 && seed_id.ieta()!=1 ) || ( seed_id.ieta()==1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()-2))){
	if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	  s4s9_[0]+=RecHitsInWindow[j].energy();
	}else{
	  if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()){
	    s4s9_[0]+=RecHitsInWindow[j].energy();
	    s4s9_[1]+=RecHitsInWindow[j].energy();
	  }else{
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
	      s4s9_[1]+=RecHitsInWindow[j].energy(); 
	    }
	  }
	}
      }else{
	if(((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()){
	  if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	    s4s9_[0]+=RecHitsInWindow[j].energy();
	    s4s9_[3]+=RecHitsInWindow[j].energy();
	  }else{
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
	      s4s9_[1]+=RecHitsInWindow[j].energy(); 
	      s4s9_[2]+=RecHitsInWindow[j].energy(); 
	    }
	  }
	}else{
	  if((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()+1 && seed_id.ieta()!=-1 ) || ( seed_id.ieta()==-1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()+2))){
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	      s4s9_[3]+=RecHitsInWindow[j].energy();
	    }else{
	      if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()){
		s4s9_[2]+=RecHitsInWindow[j].energy();
		s4s9_[3]+=RecHitsInWindow[j].energy();
	      }else{
		if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
		  s4s9_[2]+=RecHitsInWindow[j].energy(); 
		}
	      }
	    }
	  }else{
	    cout<<" (EBDetId)RecHitsInWindow[j].id()).ieta() "<<((EBDetId)RecHitsInWindow[j].id()).ieta()<<" seed_id.ieta() "<<seed_id.ieta()<<endl;
	    cout<<" Problem with S4 calculation "<<endl;return;
	  }
	}
      }
    }
    s4s9Clus.push_back(*max_element( s4s9_,s4s9_+4)/simple_energy);
    //    cout<<" s4s9Clus[0] "<<s4s9_[0]/simple_energy<<" s4s9Clus[1] "<<s4s9_[1]/simple_energy<<" s4s9Clus[2] "<<s4s9_[2]/simple_energy<<" s4s9Clus[3] "<<s4s9_[3]/simple_energy<<endl;
    //    cout<<" Max "<<*max_element( s4s9_,s4s9_+4)/simple_energy<<endl;
    nClus++;
    if (nClus == MAXCLUS) return;
  }  //  End loop over seed clusters

  // cout<< " Pi0 clusters: "<<nClus<<endl;

  // Selection, based on Simple clustering
  //pi0 candidates
  static const int MAXPI0S = 200;
  int npi0_s=0;

  vector<EBDetId> scXtals;
  scXtals.clear();

  if (nClus <= 1) return;
  for(Int_t i=0 ; i<nClus ; i++){
    for(Int_t j=i+1 ; j<nClus ; j++){
      //      cout<<" i "<<i<<"  etClus[i] "<<etClus[i]<<" j "<<j<<"  etClus[j] "<<etClus[j]<<endl;
      if( etClus[i]>selePtGammaOne_ && etClus[j]>selePtGammaTwo_ && s4s9Clus[i]>seleS4S9GammaOne_ && s4s9Clus[j]>seleS4S9GammaTwo_){
	float theta_0 = 2. * atan(exp(-etaClus[i]));
	float theta_1 = 2. * atan(exp(-etaClus[j]));
        
	float p0x = eClus[i] * sin(theta_0) * cos(phiClus[i]);
	float p1x = eClus[j] * sin(theta_1) * cos(phiClus[j]);
	float p0y = eClus[i] * sin(theta_0) * sin(phiClus[i]);
	float p1y = eClus[j] * sin(theta_1) * sin(phiClus[j]);
	float p0z = eClus[i] * cos(theta_0);
	float p1z = eClus[j] * cos(theta_1);
        
	float pt_pi0 = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
	//      cout<<" pt_pi0 "<<pt_pi0<<endl;
	if (pt_pi0 < selePtPi0_)continue;
	float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );  
	if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) ){

	  //New Loop on cluster to measure isolation:
	  vector<int> IsoClus;
	  IsoClus.clear();
	  float Iso = 0;
	  TVector3 pi0vect = TVector3((p0x+p1x), (p0y+p1y), (p0z+p1z));
	  for(Int_t k=0 ; k<nClus ; k++){
	    if(k==i || k==j)continue;
	    TVector3 Clusvect = TVector3(eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * cos(phiClus[k]), eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * sin(phiClus[k]) , eClus[k] * cos(2. * atan(exp(-etaClus[k]))));
	    float dretaclpi0 = fabs(etaClus[k] - pi0vect.Eta());
	    float drclpi0 = Clusvect.DeltaR(pi0vect);

	    if((drclpi0<selePi0BeltDR_) && (dretaclpi0<selePi0BeltDeta_) ){

	      Iso = Iso + etClus[k];
	      IsoClus.push_back(k);
	    }
	  }

	
	  if(Iso/pt_pi0<selePi0Iso_){
			
	    hMinvPi0EB_->Fill(m_inv);
	    hPt1Pi0EB_->Fill(etClus[i]);
	    hPt2Pi0EB_->Fill(etClus[j]);
	    hPtPi0EB_->Fill(pt_pi0);
	    hIsoPi0EB_->Fill(Iso/pt_pi0);
	    
		
	    npi0_s++;
	  }
          
	  if(npi0_s == MAXPI0S) return;
	}
      }
    } 
  } 

} 



void PhotonAnalyzer::endJob()
{
  
  if(standAlone_){

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
