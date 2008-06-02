#include <iostream>
//
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
//
#include "DQMOffline/EGamma/interface/PhotonAnalyzer.h"
//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaEcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
// 

 

using namespace std;

 
PhotonAnalyzer::PhotonAnalyzer( const edm::ParameterSet& pset )
  {

    fName_     = pset.getUntrackedParameter<std::string>("Name");
    verbosity_ = pset.getUntrackedParameter<int>("Verbosity");

    
    photonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
    photonCollection_ = pset.getParameter<std::string>("photonCollection");

    scBarrelProducer_       = pset.getParameter<std::string>("scBarrelProducer");
    scEndcapProducer_       = pset.getParameter<std::string>("scEndcapProducer");
    barrelEcalHits_   = pset.getParameter<edm::InputTag>("barrelEcalHits");
    endcapEcalHits_   = pset.getParameter<edm::InputTag>("endcapEcalHits");

    bcProducer_             = pset.getParameter<std::string>("bcProducer");
    bcBarrelCollection_     = pset.getParameter<std::string>("bcBarrelCollection");
    bcEndcapCollection_     = pset.getParameter<std::string>("bcEndcapCollection");

    hbheLabel_        = pset.getParameter<std::string>("hbheModule");
    hbheInstanceName_ = pset.getParameter<std::string>("hbheInstance");
    

    tracksInputTag_    = pset.getParameter<edm::InputTag>("trackProducer");   

    minPhoEtCut_ = pset.getParameter<double>("minPhoEtCut");   
    trkIsolExtRadius_ = pset.getParameter<double>("trkIsolExtR");   
    trkIsolInnRadius_ = pset.getParameter<double>("trkIsolInnR");   
    trkPtLow_     = pset.getParameter<double>("minTrackPtCut");   
    lip_       = pset.getParameter<double>("lipCut");   
    ecalIsolRadius_ = pset.getParameter<double>("ecalIsolR");   
    bcEtLow_     = pset.getParameter<double>("minBcEtCut");   
    hcalIsolExtRadius_ = pset.getParameter<double>("hcalIsolExtR");   
    hcalIsolInnRadius_ = pset.getParameter<double>("hcalIsolInnR");   
    hcalHitEtLow_     = pset.getParameter<double>("minHcalHitEtCut");   

    numOfTracksInCone_ = pset.getParameter<int>("maxNumOfTracksInCone");   
    trkPtSumCut_  = pset.getParameter<double>("trkPtSumCut");   
    ecalEtSumCut_ = pset.getParameter<double>("ecalEtSumCut");   
    hcalEtSumCut_ = pset.getParameter<double>("hcalEtSumCut");   

    parameters_ = pset;
   

}



PhotonAnalyzer::~PhotonAnalyzer() {




}


void PhotonAnalyzer::beginJob( const edm::EventSetup& setup)
{


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
  int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin"); 
  

  if (dbe_) {  
    //// All MC photons
    // SC from reco photons

    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/IsolationVariables");

    std::string histname = "nIsoTracks";    
    p_nTrackIsol_ = dbe_->bookProfile(histname,"Avg Number Of Tracks in the Iso Cone",etaBin,etaMin, etaMax,10,-0.5, 9.5);
    histname = "isoPtSum";    
    p_trackPtSum_ = dbe_->bookProfile(histname,"Avg Tracks Pt Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.);
    histname = "ecalSum";    
    p_ecalSum_ = dbe_->bookProfile(histname,"Avg Ecal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.);
    histname = "hcalSum";    
    p_hcalSum_ = dbe_->bookProfile(histname,"Avg Hcal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.);




    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/IsolatedPhotons");

    //// Reconstructed photons
    //    std::string histname = "nPho";
    histname = "nPho";
    h_nPho_[0][0] = dbe_->book1D(histname+"All","Number Of Isolated Photon candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nPho_[0][1] = dbe_->book1D(histname+"Barrel","Number Of Isolated Photon candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nPho_[0][2] = dbe_->book1D(histname+"Endcap","Number Of Isolated Photon candidates per events: Ecal Endcap ",10,-0.5, 9.5);
    histname = "scE";
    h_scE_[0][0] = dbe_->book1D(histname+"All","Isolated SC Energy: All Ecal  ",eBin,eMin, eMax);
    h_scE_[0][1] = dbe_->book1D(histname+"Barrel","Isolated SC Energy: Barrel ",eBin,eMin, eMax);
    h_scE_[0][2] = dbe_->book1D(histname+"Endcap","Isolated SC Energy: Endcap ",eBin,eMin, eMax);

    histname = "scEt";
    h_scEt_[0][0] = dbe_->book1D(histname+"All","Isolated SC Et: All Ecal ",etBin,etMin, etMax) ;
    h_scEt_[0][1] = dbe_->book1D(histname+"Barrel","Isolated SC Et: Barrel",etBin,etMin, etMax) ;
    h_scEt_[0][2] = dbe_->book1D(histname+"Endcap","Isolated SC Et: Endcap",etBin,etMin, etMax) ;

    histname = "r9";
    h_r9_[0][0] = dbe_->book1D(histname+"All",   "Isolated r9: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r9_[0][1] = dbe_->book1D(histname+"Barrel","Isolated r9: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r9_[0][2] = dbe_->book1D(histname+"Endcap","Isolated r9: Endcap ",r9Bin,r9Min, r9Max) ;

    h_scEta_[0] =   dbe_->book1D("scEta","Isolated SC Eta ",etaBin,etaMin, etaMax);
    h_scPhi_[0] =   dbe_->book1D("scPhi","Isolated SC Phi ",phiBin,phiMin,phiMax);
    h_scEtaPhi_[0]= dbe_->book2D("scEtaPhi","Isolated SC Phi vs Eta ",etaBin, etaMin, etaMax,phiBin,phiMin,phiMax);
    //
    histname = "phoE";
    h_phoE_[0][0]=dbe_->book1D(histname+"All","Isolated Photon Energy: All ecal ", eBin,eMin, eMax);
    h_phoE_[0][1]=dbe_->book1D(histname+"Barrel","Isolated Photon Energy: barrel ",eBin,eMin, eMax);
    h_phoE_[0][2]=dbe_->book1D(histname+"Endcap","Isolated Photon Energy: Endcap ",eBin,eMin, eMax);

    histname = "phoEt";
    h_phoEt_[0][0] = dbe_->book1D(histname+"All","Isolated Photon Transverse Energy: All ecal ", etBin,etMin, etMax);
    h_phoEt_[0][1] = dbe_->book1D(histname+"Barrel","Isolated Photon Transverse Energy: Barrel ",etBin,etMin, etMax);
    h_phoEt_[0][2] = dbe_->book1D(histname+"Endcap","Isolated Photon Transverse Energy: Endcap ",etBin,etMin, etMax);

    h_phoEta_[0] = dbe_->book1D("phoEta","Isolated Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoPhi_[0] = dbe_->book1D("phoPhi","Isolated Photon  Phi ",phiBin,phiMin,phiMax) ;

    histname="nConv";
    h_nConv_[0][0] = dbe_->book1D(histname+"All","Number Of Conversions per isolated candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nConv_[0][1] = dbe_->book1D(histname+"Barrel","Number Of Conversions per isolated candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nConv_[0][2] = dbe_->book1D(histname+"Endcap","Number Of Conversions per isolated candidates per events: Ecal Endcap ",10,-0.5, 9.5);

    h_convEta_[0] = dbe_->book1D("convEta","Isolated converted Photon Eta ",etaBin,etaMin, etaMax) ;
    h_convPhi_[0] = dbe_->book1D("convPhi","Isolated converted Photon  Phi ",phiBin,phiMin,phiMax) ;

    histname="r9VsTracks";
    h_r9VsNofTracks_[0][0] = dbe_->book2D(histname+"All","Isolated photons r9 vs nTracks from conversions: All Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][1] = dbe_->book2D(histname+"Barrel","Isolated photons r9 vs nTracks from conversions: Barrel Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;
    h_r9VsNofTracks_[0][2] = dbe_->book2D(histname+"Endcap","Isolated photons r9 vs nTracks from conversions: Endcap Ecal",r9Bin,r9Min, r9Max, 3, -0.5, 2.5) ;

    histname="EoverPtracks";
    h_EoverPTracks_[0][0] = dbe_->book1D(histname+"All","Isolated photons conversion E/p: all Ecal ",100, 0., 5.);
    h_EoverPTracks_[0][1] = dbe_->book1D(histname+"Barrel","Isolated photons conversion E/p: Barrel Ecal",100, 0., 5.);
    h_EoverPTracks_[0][2] = dbe_->book1D(histname+"All","Isolated photons conversion E/p: Endcap Ecal ",100, 0., 5.);

    histname="pTknHitsVsEta";
    // p_tk_nHitsVsEta_[0] =  dbe_->bookProfile(histname,"Isolated Photons:Tracks from conversions: mean numb of  Hits vs Eta",etaBin,etaMin, etaMax);
    p_tk_nHitsVsEta_[0] =  dbe_->bookProfile(histname,histname,etaBin,etaMin, etaMax,etaBin,0, 16);
    h_tkChi2_[0] = dbe_->book1D("tkChi2","Isolated Photons:Tracks from conversions: #chi^{2} of tracks", 100, 0., 20.0); 

    histname="hDPhiTracksAtVtx";
    h_DPhiTracksAtVtx_[0][0] =dbe_->book1D(histname+"All", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[0][1] =dbe_->book1D(histname+"Barrel", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[0][2] =dbe_->book1D(histname+"Endcap", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    histname="hDCotTracks";
    h_DCotTracks_[0][0]= dbe_->book1D(histname+"All","Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DCotTracks_[0][1]= dbe_->book1D(histname+"Barrel","Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DCotTracks_[0][2]= dbe_->book1D(histname+"Encap","Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    histname="hInvMass";
    h_invMass_[0][0]= dbe_->book1D(histname+"All","Isolated Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[0][1]= dbe_->book1D(histname+"Barrel","Isolated Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[0][2]= dbe_->book1D(histname+"Endcap","Isolated Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);
    histname="hDPhiTracksAtEcal";
    h_DPhiTracksAtEcal_[0][0]= dbe_->book1D(histname+"All","Isolated Photons:Tracks from conversions:  #delta#phi at Ecal : all Ecal ",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtEcal_[0][1]= dbe_->book1D(histname+"Barrel","Isolated Photons:Tracks from conversions:  #delta#phi at Ecal : Barrel Ecal ",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtEcal_[0][2]= dbe_->book1D(histname+"Endcap","Isolated Photons:Tracks from conversions:  #delta#phi at Ecal : Endcap Ecal ",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    histname="hDEtaTracksAtEcal";
    h_DEtaTracksAtEcal_[0][0]= dbe_->book1D(histname+"All","Isolated Photons:Tracks from conversions:  #delta#eta at Ecal : all Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DEtaTracksAtEcal_[0][1]= dbe_->book1D(histname+"Barrel","Isolated Photons:Tracks from conversions:  #delta#eta at Ecal : Barrel Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
    h_DEtaTracksAtEcal_[0][2]= dbe_->book1D(histname+"Endcap","Isolated Photons:Tracks from conversions:  #delta#eta at Ecal : Endcap Ecal ",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax); 
   
    h_convVtxRvsZ_[0] =   dbe_->book2D("convVtxRvsZ","Isolated Photon Reco conversion vtx position",100, 0., 280.,200,0., 120.);
    h_zPVFromTracks_[0] =  dbe_->book1D("zPVFromTracks","Isolated Photons: PV z from conversion tracks",100, -25., 25.);


    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer/NonIsolatedPhotons");


    histname = "nPhoNoIs"; 
    h_nPho_[1][0] = dbe_->book1D(histname+"All","Number Of Non Isolated Photon candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nPho_[1][1] = dbe_->book1D(histname+"Barrel","Number Of Non Isolated Photon candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nPho_[1][2] = dbe_->book1D(histname+"Endcap","Number Of Non Isolated Photon candidates per events: Ecal Endcap ",10,-0.5, 9.5);


    histname = "scENoIs";
    h_scE_[1][0] = dbe_->book1D(histname+"All","Non Isolated SC Energy: All Ecal  ",eBin,eMin, eMax);
    h_scE_[1][1] = dbe_->book1D(histname+"Barrel","Non Isolated SC Energy: Barrel ",eBin,eMin, eMax);
    h_scE_[1][2] = dbe_->book1D(histname+"Endcap","Non Isolated SC Energy: Endcap ",eBin,eMin, eMax);


    histname = "scEtNoIs";
    h_scEt_[1][0] = dbe_->book1D(histname+"All","Non Isolated SC Et: All Ecal ",etBin,etMin, etMax) ;
    h_scEt_[1][1] = dbe_->book1D(histname+"Barrel","Non Isolated SC Et: Barrel",etBin,etMin, etMax) ;
    h_scEt_[1][2] = dbe_->book1D(histname+"Endcap","Non Isolated SC Et: Endcap",etBin,etMin, etMax) ;

    histname = "r9NoIs";
    h_r9_[1][0] = dbe_->book1D(histname+"All",   "Non Isolated r9: All Ecal",r9Bin,r9Min, r9Max) ;
    h_r9_[1][1] = dbe_->book1D(histname+"Barrel","Non Isolated r9: Barrel ",r9Bin,r9Min, r9Max) ;
    h_r9_[1][2] = dbe_->book1D(histname+"Endcap","Non Isolated r9: Endcap ",r9Bin,r9Min, r9Max) ;

    h_scEta_[1] =   dbe_->book1D("scEtaNoIs","Non Isolated SC Eta ",etaBin,etaMin, etaMax);
    h_scPhi_[1] =   dbe_->book1D("scPhiNois","Non Isolated SC Phi ",phiBin,phiMin,phiMax);
    h_scEtaPhi_[1]= dbe_->book2D("scEtaPhiNoIs","Non Isolated SC Phi vs Eta ",etaBin, etaMin, etaMax,phiBin,phiMin,phiMax);

    //
    histname = "phoENoIs";
    h_phoE_[1][0]=dbe_->book1D(histname+"All","Non Isolated Photon Energy: All ecal ", eBin,eMin, eMax);
    h_phoE_[1][1]=dbe_->book1D(histname+"Barrel","Non Isolated Photon Energy: Barrel ",eBin,eMin, eMax);
    h_phoE_[1][2]=dbe_->book1D(histname+"Endcap","Non Isolated Photon Energy: Endcap ",eBin,eMin, eMax);

    histname = "phoEtNoIs";
    h_phoEt_[1][0] = dbe_->book1D(histname+"All","Non Isolated Photon Transverse Energy: All ecal ", etBin,etMin, etMax);
    h_phoEt_[1][1] = dbe_->book1D(histname+"Barrel","Non Isolated Photon Transverse Energy: Barrel ",etBin,etMin, etMax);
    h_phoEt_[1][2] = dbe_->book1D(histname+"Endcap","Non Isolated Photon Transverse Energy: Endcap ",etBin,etMin, etMax);

    h_phoEta_[1] = dbe_->book1D("phoEtaNoIs","Non Isolated Photon Eta ",etaBin,etaMin, etaMax) ;
    h_phoPhi_[1] = dbe_->book1D("phoPhiNoIs","Non Isolated Photon  Phi ",phiBin,phiMin,phiMax) ;

    histname="nConvNoIs";
    h_nConv_[1][0] = dbe_->book1D(histname+"All","Number Of Conversions per non isolated candidates per events: All Ecal  ",10,-0.5, 9.5);
    h_nConv_[1][1] = dbe_->book1D(histname+"Barrel","Number Of Conversions per non isolated candidates per events: Ecal Barrel  ",10,-0.5, 9.5);
    h_nConv_[1][2] = dbe_->book1D(histname+"Endcap","Number Of Conversions per non isolated candidates per events: Ecal Endcap ",10,-0.5, 9.5);

    h_convEta_[1] = dbe_->book1D("convEtaNoIs","Non Isolated converted Photon Eta ",etaBin,etaMin, etaMax) ;
    h_convPhi_[1] = dbe_->book1D("convPhiNoIs","Non Isolated converted Photon  Phi ",phiBin,phiMin,phiMax) ;

    histname="r9VsTracksNoIs";
    h_r9VsNofTracks_[1][0] = dbe_->book2D(histname+"All","Non Isolated photons r9 vs nTracks from conversions: All Ecal",r9Bin,r9Min, r9Max,3, -0.5, 2.5) ;
    h_r9VsNofTracks_[1][1] = dbe_->book2D(histname+"Barrel","Non Isolated photons r9 vs nTracks from conversions: Barrel Ecal",r9Bin,r9Min, r9Max,3, -0.5, 2.5) ;
    h_r9VsNofTracks_[1][2] = dbe_->book2D(histname+"Endcap","Non Isolated photons r9 vs nTracks from conversions: Endcap Ecal",r9Bin,r9Min, r9Max,3, -0.5, 2.5) ;
    histname="EoverPtracksNoIs";
    h_EoverPTracks_[1][0] = dbe_->book1D(histname+"All","Non Isolated photons conversion E/p: all Ecal ",100, 0., 5.);
    h_EoverPTracks_[1][1] = dbe_->book1D(histname+"Barrel","Non Isolated photons conversion E/p: Barrel Ecal",100, 0., 5.);
    h_EoverPTracks_[1][2] = dbe_->book1D(histname+"All","Non Isolated photons conversion E/p: Endcap Ecal ",100, 0., 5.);

    p_tk_nHitsVsEta_[1] =  dbe_->bookProfile("pTknHitsVsEtaNoIs","Non Isolated Photons:Tracks from conversions: mean numb of  Hits vs Eta",etaBin,etaMin, etaMax,etaBin,0, 16);
    h_tkChi2_[1] = dbe_->book1D("tkChi2NoIs","NonIsolated Photons:Tracks from conversions: #chi^{2} of tracks", 100, 0., 20.0); 
    histname="hDPhiTracksAtVtxNoIs";
    h_DPhiTracksAtVtx_[1][0] =dbe_->book1D(histname+"All", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[1][1] =dbe_->book1D(histname+"Barrel", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    h_DPhiTracksAtVtx_[1][2] =dbe_->book1D(histname+"Endcap", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax); 
    histname="hDCotTracksNoIs";
    h_DCotTracks_[1][0]= dbe_->book1D(histname+"All","Non Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Eca ",100, -0.2, 0.2);
    h_DCotTracks_[1][1]= dbe_->book1D(histname+"Barrel","Non Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal ",100, -0.2, 0.2);
    h_DCotTracks_[1][2]= dbe_->book1D(histname+"Encap","Non Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Eca ",100, -0.2, 0.2);
    histname="hInvMassNoIs";
    h_invMass_[1][0]= dbe_->book1D(histname+"All","Non Isolated Photons:Tracks from conversion: Pair invariant mass: all Ecal ",100, 0., 1.5);
    h_invMass_[1][1]= dbe_->book1D(histname+"Barrel","Non Isolated Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal ",100, 0., 1.5);
    h_invMass_[1][2]= dbe_->book1D(histname+"Endcap","Non Isolated Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal ",100, 0., 1.5);
    histname="hDPhiTracksAtEcalNoIs";
    h_DPhiTracksAtEcal_[1][0]= dbe_->book1D(histname+"All","Non Isolated Photons:Tracks from conversions: #delta#phi at Ecal : all Ecal ",100, -0.2, 0.2);
    h_DPhiTracksAtEcal_[1][1]= dbe_->book1D(histname+"Barrel","Non Isolated Photons:Tracks from conversions: #delta#phi at Ecal : Barrel Ecal ",100, -0.2, 0.2);
    h_DPhiTracksAtEcal_[1][2]= dbe_->book1D(histname+"Endcap","Non Isolated Photons:Tracks from conversions: #delta#phi at Ecal : Endcap Ecal ",100, -0.2, 0.2);
    histname="hDEtaTracksAtEcalNoIs";
    h_DEtaTracksAtEcal_[1][0]= dbe_->book1D(histname+"All","Non Isolated Photons:Tracks from conversions: #delta#eta at Ecal : all Ecal ",100, -0.2, 0.2);
    h_DEtaTracksAtEcal_[1][1]= dbe_->book1D(histname+"Barrel","Non Isolated Photons:Tracks from conversions: #delta#eta at Ecal : Barrel Ecal ",100, -0.2, 0.2);
    h_DEtaTracksAtEcal_[1][2]= dbe_->book1D(histname+"Endcap","Non Isolated Photons:Tracks from conversions: #delta#eta at Ecal : Endcap Ecal ",100, -0.2, 0.2);
 

    h_convVtxRvsZ_[1] =  dbe_->book2D("convVtxRvsZNoIs","Non Isolated Photon Reco conversion vtx position",100, 0., 280.,200,0., 120.);
    h_zPVFromTracks_[1] =  dbe_->book1D("zPVFromTracks","Non Isolated Photons: PV z from conversion tracks",100, -25., 25.);

  }

  
  return ;
}





void PhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  
  
  using namespace edm;
  const float etaPhiDistance=0.01;
  // Fiducial region
  const float TRK_BARL =0.9;
  const float BARL = 1.4442; // DAQ TDR p.290
  const float END_LO = 1.566;
  const float END_HI = 2.5;
 // Electron mass
  const Float_t mElec= 0.000511;


  nEvt_++;  
  LogInfo("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
  //  LogDebug("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "PhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
 
  
  ///// Get the recontructed  photons
  Handle<reco::PhotonCollection> photonHandle; 
  e.getByLabel(photonCollectionProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
  std::cout  << "PhotonAnalyzer  Photons with conversions collection size " << photonCollection.size() << "\n";

  
 // Get EcalRecHits
  Handle<EcalRecHitCollection> barrelHitHandle;
  e.getByLabel(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
    return;
  }
  const EcalRecHitCollection *barrelRecHits = barrelHitHandle.product();


  Handle<EcalRecHitCollection> endcapHitHandle;
  e.getByLabel(endcapEcalHits_, endcapHitHandle);
  if (!endcapHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
    return;
  }
  const EcalRecHitCollection *endcapRecHits = endcapHitHandle.product();

  // get the geometry from the event setup:
  esup.get<IdealGeometryRecord>().get(theCaloGeom_);

  Handle<HBHERecHitCollection> hbhe;
  std::auto_ptr<HBHERecHitMetaCollection> mhbhe;
  e.getByLabel(hbheLabel_,hbheInstanceName_,hbhe);  
  if (!hbhe.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<hbheInstanceName_.c_str();
    return; 
  }

  mhbhe=  std::auto_ptr<HBHERecHitMetaCollection>(new HBHERecHitMetaCollection(*hbhe));

  // Get the tracks
  edm::Handle<reco::TrackCollection> tracksHandle;
  e.getByLabel(tracksInputTag_,tracksHandle);
  const reco::TrackCollection* trackCollection = tracksHandle.product();

  std::vector<int> nPho(2);
  std::vector<int> nPhoBarrel(2);
  std::vector<int> nPhoEndcap(2);
  for ( int i=0; i<nPho.size(); i++ ) nPho[i]=0;
  for ( int i=0; i<nPhoBarrel.size(); i++ ) nPhoBarrel[i]=0;
  for ( int i=0; i<nPhoEndcap.size(); i++ ) nPhoEndcap[i]=0;


  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {


    if ( (*iPho).energy()/ cosh( (*iPho).eta()) < minPhoEtCut_) continue; 

    int detector=0;
    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;
    float etaPho=(*iPho).phi();
    float phiPho=(*iPho).eta();
    if ( fabs(etaPho) <  1.479 ) {
      phoIsInBarrel=true;
    } else {
      phoIsInEndcap=true;
    }

    float phiClu=(*iPho).superCluster()->phi();
    float etaClu=(*iPho).superCluster()->eta();

    bool  scIsInBarrel=false;
    bool  scIsInEndcap=false;
    if ( fabs(etaClu) <  1.479 ) 
      scIsInBarrel=true;
    else
      scIsInEndcap=true;
    
    
    int nTracks=0;
    double ptSum=0.;
    double ecalSum=0.;
    double hcalSum=0.;
    /// isolation in the tracker
    PhotonTkIsolation trackerIsol(trkIsolExtRadius_, trkIsolInnRadius_, trkPtLow_, lip_, trackCollection);     
    nTracks = trackerIsol.getNumberTracks(&(*iPho));
    ptSum = trackerIsol.getPtTracks(&(*iPho));


    /// isolation in Ecal
    edm::Handle<reco::BasicClusterCollection> bcHandle;
    edm::Handle<reco::SuperClusterCollection> scHandle;
    if ( phoIsInBarrel ) {
      // Get the basic cluster collection in the Barrel 
      e.getByLabel(bcProducer_, bcBarrelCollection_, bcHandle);
      if (!bcHandle.isValid()) {
	edm::LogError("ConversionTrackCandidateProducer") << "Error! Can't get the product "<<bcBarrelCollection_.c_str();
	return;
      }

      // Get the  Barrel Super Cluster collection
      e.getByLabel(scBarrelProducer_,scHandle);
      if (!scHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<scBarrelProducer_.label();
	return;
      }

    } else if ( phoIsInEndcap ) {    
      // Get the basic cluster collection in the Endcap 
      e.getByLabel(bcProducer_, bcEndcapCollection_, bcHandle);
      if (!bcHandle.isValid()) {
	edm::LogError("CoonversionTrackCandidateProducer") << "Error! Can't get the product "<<bcEndcapCollection_.c_str();
	return;
      }


      // Get the  Endcap Super Cluster collection
      e.getByLabel(scEndcapProducer_,scHandle);
      if (!scHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<scEndcapProducer_.label();
	return;
      }


    }

    const reco::SuperClusterCollection scCollection = *(scHandle.product());
    const reco::BasicClusterCollection bcCollection = *(bcHandle.product());
    
    EgammaEcalIsolation ecalIsol( ecalIsolRadius_, bcEtLow_, &bcCollection, &scCollection);
    ecalSum = ecalIsol.getEcalEtSum(&(*iPho));
    /// isolation in Hcal
    EgammaHcalIsolation hcalIsol (hcalIsolExtRadius_,hcalIsolInnRadius_,hcalHitEtLow_,theCaloGeom_.product(),mhbhe.get()); 
    hcalSum = hcalIsol.getHcalEtSum(&(*iPho));

    bool isIsolated=false;

    if ( (nTracks < numOfTracksInCone_) && 
	 ( ptSum < trkPtSumCut_) &&
	 ( ecalSum < ecalEtSumCut_ ) &&
	 ( hcalSum < hcalEtSumCut_ ) ) isIsolated = true;

    p_nTrackIsol_->Fill( (*iPho).superCluster()->eta(), float(nTracks));
    p_trackPtSum_->Fill((*iPho).superCluster()->eta(), ptSum);
    p_ecalSum_->Fill((*iPho).superCluster()->eta(), ecalSum);
    p_hcalSum_->Fill((*iPho).superCluster()->eta(), hcalSum);




    int type=0;
    if ( !isIsolated ) type=1;


    nEntry_++;

    nPho[type]++; 
    if (phoIsInBarrel) nPhoBarrel[type]++;
    if (phoIsInEndcap) nPhoEndcap[type]++;
    
    
    h_scEta_[type]->Fill( (*iPho).superCluster()->eta() );
    h_scPhi_[type]->Fill( (*iPho).superCluster()->phi() );
    h_scEtaPhi_[type]->Fill( (*iPho).superCluster()->eta(),(*iPho).superCluster()->phi() );

    h_scE_[type][0]->Fill( (*iPho).superCluster()->energy() );
    h_scEt_[type][0]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->eta()) );
    h_r9_[type][0]->Fill( (*iPho).r9() );

    h_phoEta_[type]->Fill( (*iPho).eta() );
    h_phoPhi_[type]->Fill( (*iPho).phi() );
    
    h_phoE_[type][0]->Fill( (*iPho).energy() );
    h_phoEt_[type][0]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );

    h_nConv_[type][0]->Fill(float( (*iPho).conversions().size()));


    if ( scIsInBarrel ) {
      h_scE_[type][1]->Fill( (*iPho).superCluster()->energy() );
      h_scEt_[type][1]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->eta()) );
      h_r9_[type][1]->Fill( (*iPho).r9() );
    }
    if ( scIsInEndcap ) {
      h_scE_[type][2]->Fill( (*iPho).superCluster()->energy() );
      h_scEt_[type][2]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->eta()) );
      h_r9_[type][2]->Fill( (*iPho).r9() );
    }



    if ( phoIsInBarrel ) {
      h_phoE_[type][1]->Fill( (*iPho).energy() );
      h_phoEt_[type][1]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
      h_nConv_[type][1]->Fill(float( (*iPho).conversions().size()));
    }
    
    if ( phoIsInEndcap ) {
      h_phoE_[type][2]->Fill( (*iPho).energy() );
      h_phoEt_[type][2]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
      h_nConv_[type][2]->Fill(float( (*iPho).conversions().size()));
    }
    
    ////////////////// plot quantitied related to conversions
    std::vector<reco::ConversionRef> conversions = (*iPho).conversions();
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

      h_r9VsNofTracks_[type][0]->Fill( (*iPho).r9(), conversions[iConv]->nTracks() ) ; 
      if ( phoIsInBarrel ) h_r9VsNofTracks_[type][1]->Fill( (*iPho).r9(), conversions[iConv]->nTracks() ) ; 
      if ( phoIsInEndcap ) h_r9VsNofTracks_[type][2]->Fill( (*iPho).r9(), conversions[iConv]->nTracks() ) ; 

      if ( conversions[iConv]->nTracks() <2 ) continue; 


      h_convEta_[type]->Fill( conversions[iConv]->superCluster()->eta() );
      h_convPhi_[type]->Fill( conversions[iConv]->superCluster()->phi() );
      h_EoverPTracks_[type][0] ->Fill( conversions[iConv]->EoverP() ) ;
      if ( phoIsInBarrel ) h_EoverPTracks_[type][1] ->Fill( conversions[iConv]->EoverP() ) ;
      if ( phoIsInEndcap ) h_EoverPTracks_[type][2] ->Fill( conversions[iConv]->EoverP() ) ;


      //      h_invMass_[type][0] ->Fill(  conversions[iConv]->pairInvariantMass()); 
      //if ( phoIsInBarrel ) h_invMass_[type][1] ->Fill(conversions[iConv]->pairInvariantMass()); 
      //if ( phoIsInEndcap ) h_invMass_[type][2] ->Fill(conversions[iConv]->pairInvariantMass()); 


      
      if ( conversions[iConv]->conversionVertex().isValid() ) 
	h_convVtxRvsZ_[type] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	

      h_zPVFromTracks_[type]->Fill ( conversions[iConv]->zOfPrimaryVertexFromTracks() );

      std::vector<reco::TrackRef> tracks = conversions[iConv]->tracks();



      float px=0;
      float py=0;
      float pz=0;
      float e=0;
      for (unsigned int i=0; i<tracks.size(); i++) {
	p_tk_nHitsVsEta_[type]->Fill(  conversions[iConv]->superCluster()->eta(),   float(tracks[i]->recHitsSize() ) );
	h_tkChi2_[type] ->Fill (tracks[i]->normalizedChi2() ); 
        px+= tracks[i]->innerMomentum().x();
        py+= tracks[i]->innerMomentum().y();
        pz+= tracks[i]->innerMomentum().z();
        e +=  sqrt (  tracks[i]->innerMomentum().x()*tracks[i]->innerMomentum().x() +
		      tracks[i]->innerMomentum().y()*tracks[i]->innerMomentum().y() +
		      tracks[i]->innerMomentum().z()*tracks[i]->innerMomentum().z() +
		      +  mElec*mElec ) ;
      }
      float totP = sqrt(px*px +py*py + pz*pz);
      float invM=  (e + totP) * (e-totP) ;

      if ( invM> 0.) {
	invM= sqrt( invM);
      } else {
	invM=-1;
      }

      h_invMass_[type][0] ->Fill( invM);
      if ( phoIsInBarrel ) h_invMass_[type][1] ->Fill(invM);
      if ( phoIsInEndcap ) h_invMass_[type][2] ->Fill(invM);


      
      float  dPhiTracksAtVtx = -99;
      
      float phiTk1= tracks[0]->innerMomentum().phi();
      float phiTk2= tracks[1]->innerMomentum().phi();
      dPhiTracksAtVtx = phiTk1-phiTk2;
      dPhiTracksAtVtx = phiNormalization( dPhiTracksAtVtx );
      h_DPhiTracksAtVtx_[type][0]->Fill( dPhiTracksAtVtx);
      if ( phoIsInBarrel ) h_DPhiTracksAtVtx_[type][1]->Fill( dPhiTracksAtVtx);
      if ( phoIsInEndcap ) h_DPhiTracksAtVtx_[type][2]->Fill( dPhiTracksAtVtx);
      h_DCotTracks_[type][0] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );
      if ( phoIsInBarrel ) h_DCotTracks_[type][1] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );
      if ( phoIsInEndcap ) h_DCotTracks_[type][2] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );
      
      
      float  dPhiTracksAtEcal=-99;
      float  dEtaTracksAtEcal=-99;
      if (conversions[iConv]-> bcMatchingWithTracks()[0].isNonnull() && conversions[iConv]->bcMatchingWithTracks()[1].isNonnull() ) {
	
	
	float recoPhi1 = conversions[iConv]->ecalImpactPosition()[0].phi();
	float recoPhi2 = conversions[iConv]->ecalImpactPosition()[1].phi();
	float recoEta1 = conversions[iConv]->ecalImpactPosition()[0].eta();
	float recoEta2 = conversions[iConv]->ecalImpactPosition()[1].eta();
	float bcPhi1 = conversions[iConv]->bcMatchingWithTracks()[0]->phi();
	float bcPhi2 = conversions[iConv]->bcMatchingWithTracks()[1]->phi();
	float bcEta1 = conversions[iConv]->bcMatchingWithTracks()[0]->eta();
	float bcEta2 = conversions[iConv]->bcMatchingWithTracks()[1]->eta();
	recoPhi1 = phiNormalization(recoPhi1);
	recoPhi2 = phiNormalization(recoPhi2);
	bcPhi1 = phiNormalization(bcPhi1);
	bcPhi2 = phiNormalization(bcPhi2);
	dPhiTracksAtEcal = recoPhi1 -recoPhi2;
	dPhiTracksAtEcal = phiNormalization( dPhiTracksAtEcal );
	dEtaTracksAtEcal = recoEta1 -recoEta2;
	
	h_DPhiTracksAtEcal_[type][0]->Fill( dPhiTracksAtEcal);
	h_DEtaTracksAtEcal_[type][0]->Fill( dEtaTracksAtEcal);
	if ( phoIsInBarrel ) {
	  h_DPhiTracksAtEcal_[type][1]->Fill( dPhiTracksAtEcal);
	  h_DEtaTracksAtEcal_[type][1]->Fill( dEtaTracksAtEcal);
	}
	if ( phoIsInEndcap ) {
	  h_DPhiTracksAtEcal_[type][2]->Fill( dPhiTracksAtEcal);
	  h_DEtaTracksAtEcal_[type][2]->Fill( dEtaTracksAtEcal);
	}
	
      }
      
			      
    } // loop over conversions
    
    
    
  }/// End loop over Reco  particles
    


  h_nPho_[0][0]-> Fill (float(nPho[0]));
  h_nPho_[0][1]-> Fill (float(nPhoBarrel[0]));
  h_nPho_[0][2]-> Fill (float(nPhoEndcap[0]));
  h_nPho_[1][0]-> Fill (float(nPho[1]));
  h_nPho_[1][1]-> Fill (float(nPhoBarrel[1]));
  h_nPho_[1][2]-> Fill (float(nPhoEndcap[1]));


  

}




void PhotonAnalyzer::endJob()
{



  dbe_->showDirStructure();
  bool outputMEsInRootFile = parameters_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
  
  edm::LogInfo("PhotonAnalyzer") << "Analyzed " << nEvt_  << "\n";
  // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  std::cout  << "PhotonAnalyzer::endJob Analyzed " << nEvt_ << " events " << "\n";
  std::cout << " Total number of photons " << nEntry_ << std::endl;
   
  return ;
}
 
float PhotonAnalyzer::phiNormalization(float & phi)
{
//---Definitions
 const float PI    = 3.1415927;
 const float TWOPI = 2.0*PI;


 if(phi >  PI) {phi = phi - TWOPI;}
 if(phi < -PI) {phi = phi + TWOPI;}

 //  cout << " Float_t PHInormalization out " << PHI << endl;
 return phi;

}



