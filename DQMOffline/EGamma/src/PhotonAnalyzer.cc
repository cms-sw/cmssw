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

  if (dbe_) {  
    //// All MC photons
    // SC from reco photons

    dbe_->setCurrentFolder("Egamma/PhotonAnalyzer");

    //// Reconstructed photons
    std::string histname = "nPho";
    h_nPho_.push_back(dbe_->book1D(histname+"all","Numbef Of Isolated Photon candidates per events: all Ecal  ",10,-0.5, 9.5));
    h_nPho_.push_back(dbe_->book1D(histname+"barrel","Numbef Of Isolated Photon candidates per events: Ecal Barrel  ",10,-0.5, 9.5));
    h_nPho_.push_back(dbe_->book1D(histname+"endcap","Numbef Of Isolated Photon candidates per events: Ecal Endcap ",10,-0.5, 9.5));
    histname = "scE";
    h_scE_.push_back(dbe_->book1D(histname+"all","SC Energy: all Ecal  ",eBin,eMin, eMax));
    h_scE_.push_back(dbe_->book1D(histname+"barrel","SC Energy: Barrel ",eBin,eMin, eMax));
    h_scE_.push_back(dbe_->book1D(histname+"endcap","SC Energy: Endcap ",eBin,eMin, eMax));

    histname = "scEt";
    h_scEt_.push_back( dbe_->book1D(histname+"all","SC Et: all Ecal ",etBin,etMin, etMax) );
    h_scEt_.push_back( dbe_->book1D(histname+"barrel","SC Et: Barrel",etBin,etMin, etMax) );
    h_scEt_.push_back( dbe_->book1D(histname+"endcap","SC Et: Endcap",etBin,etMin, etMax) );

    histname = "r9";
    h_r9_.push_back( dbe_->book1D(histname+"all",   "r9: all Ecal",r9Bin,r9Min, r9Max) );
    h_r9_.push_back( dbe_->book1D(histname+"barrel","r9: all Ecal",r9Bin,r9Min, r9Max) );
    h_r9_.push_back( dbe_->book1D(histname+"endcap","r9: all Ecal",r9Bin,r9Min, r9Max) );

    h_scEta_ = dbe_->book1D("scEta","SC Eta ",etaBin,etaMin, etaMax);
    h_scPhi_ = dbe_->book1D("scPhi","SC Phi ",phiBin,phiMin,phiMax);
    h_scEtaPhi_ = dbe_->book2D("scEtaPhi","SC Phi vs Eta ",etaBin, etaMin, etaMax,phiBin,phiMin,phiMax);
    //
    histname = "phoE";
    h_phoE_.push_back(dbe_->book1D(histname+"all","Photon Energy: all ecal ", eBin,eMin, eMax));
    h_phoE_.push_back(dbe_->book1D(histname+"barrel","Photon Energy: barrel ",eBin,eMin, eMax));
    h_phoE_.push_back(dbe_->book1D(histname+"endcap","Photon Energy: endcap ",eBin,eMin, eMax));
    histname = "phoEt";
    h_phoEt_.push_back(dbe_->book1D(histname+"all","Photon Transverse Energy: all ecal ", etBin,etMin, etMax));
    h_phoEt_.push_back(dbe_->book1D(histname+"barrel","Photon Transverse Energy: barrel ",etBin,etMin, etMax));
    h_phoEt_.push_back(dbe_->book1D(histname+"endcap","Photon Transverse Energy: endcap ",etBin,etMin, etMax));

    h_phoEta_ = dbe_->book1D("phoEta","Photon Eta ",etaBin,etaMin, etaMax);
    h_phoPhi_ = dbe_->book1D("phoPhi","Photon  Phi ",phiBin,phiMin,phiMax);
    
     // SC from reco photons
    /*
    dbe_->tag( h_scE_->getFullname(),10);
    dbe_->tag( h_scEt_->getFullname(),11);
    dbe_->tag( h_scEta_->getFullname(),12);
    dbe_->tag( h_scPhi_->getFullname(),13);
    
    dbe_->tag( h_phoE_->getFullname(),14);
    dbe_->tag( h_phoEta_->getFullname(),15);
    dbe_->tag( h_phoPhi_->getFullname(),16);
    
    */

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
  

  int nPho=0;
  int nPhoBarrel=0;
  int nPhoEndcap=0;  
  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {


    if ( (*iPho).energy()/ cosh( (*iPho).eta()) < minPhoEtCut_) continue; 

    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;
    float etaPho=(*iPho).phi();
    float phiPho=(*iPho).eta();
    if ( fabs(etaPho) <  1.479 ) 
      phoIsInBarrel=true;
    else
      phoIsInEndcap=true;

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
	edm::LogError("ConverionTrackCandidateProducer") << "Error! Can't get the product "<<bcBarrelCollection_.c_str();
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


    nEntry_++;

    if ( isIsolated ) {
      nPho++;
      if (phoIsInBarrel)  nPhoBarrel++;
      if (phoIsInEndcap)  nPhoEndcap++;
      
      
      
      h_scEta_->Fill( (*iPho).superCluster()->position().eta() );
      h_scPhi_->Fill( (*iPho).superCluster()->position().phi() );
      h_scEtaPhi_->Fill( (*iPho).superCluster()->position().eta(),(*iPho).superCluster()->position().phi() );
      
      
      h_scE_[0]->Fill( (*iPho).superCluster()->energy() );
      h_scEt_[0]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->position().eta()) );
      h_r9_[0]->Fill( (*iPho).r9() );
      
      if ( scIsInBarrel ) {
	h_scE_[1]->Fill( (*iPho).superCluster()->energy() );
	h_scEt_[1]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->position().eta()) );
      }
      if ( scIsInEndcap ) {
	h_scE_[2]->Fill( (*iPho).superCluster()->energy() );
	h_scEt_[2]->Fill( (*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->position().eta()) );
      }
      
      
      h_phoEta_->Fill( (*iPho).eta() );
      h_phoPhi_->Fill( (*iPho).phi() );
      
      h_phoE_[0]->Fill( (*iPho).energy() );
      h_phoEt_[0]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
      h_r9_[0]->Fill( (*iPho).r9());
      
      if ( phoIsInBarrel ) {
	h_phoE_[1]->Fill( (*iPho).energy() );
	h_phoEt_[1]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
	h_r9_[1]->Fill( (*iPho).r9());
      }
      
      if ( phoIsInEndcap ) {
	h_phoE_[2]->Fill( (*iPho).energy() );
	h_phoEt_[2]->Fill( (*iPho).energy()/ cosh( (*iPho).eta()) );
	h_r9_[2]->Fill( (*iPho).r9());
      }
      
    }
      
  }/// End loop over Reco  particles
    


  h_nPho_[0]-> Fill (float(nPho));
  h_nPho_[1]-> Fill (float(nPhoBarrel));
  h_nPho_[2]-> Fill (float(nPhoEndcap));


  

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
 


