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
#include "DataFormats/EgammaCandidates/interface/PhotonIDFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/PhotonIDAssociation.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaEcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
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

    fName_              = pset.getUntrackedParameter<std::string>("Name");
    verbosity_          = pset.getUntrackedParameter<int>("Verbosity");

    
    photonProducer_     = pset.getParameter<std::string>("phoProducer");
    photonCollection_   = pset.getParameter<std::string>("photonCollection");

    scBarrelProducer_   = pset.getParameter<edm::InputTag>("scBarrelProducer");
    scEndcapProducer_   = pset.getParameter<edm::InputTag>("scEndcapProducer");

    barrelEcalHits_     = pset.getParameter<edm::InputTag>("barrelEcalHits");
    endcapEcalHits_     = pset.getParameter<edm::InputTag>("endcapEcalHits");

    bcBarrelProducer_   = pset.getParameter<std::string>("bcBarrelProducer");
    bcEndcapProducer_   = pset.getParameter<std::string>("bcEndcapProducer");      
    bcBarrelCollection_ = pset.getParameter<std::string>("bcBarrelCollection");
    bcEndcapCollection_ = pset.getParameter<std::string>("bcEndcapCollection");

    hbheLabel_          = pset.getParameter<std::string>("hbheModule");
    hbheInstanceName_   = pset.getParameter<std::string>("hbheInstance");

    tracksInputTag_     = pset.getParameter<edm::InputTag>("trackProducer");   

    minPhoEtCut_        = pset.getParameter<double>("minPhoEtCut");   
    trkIsolExtRadius_   = pset.getParameter<double>("trkIsolExtR");   
    trkIsolInnRadius_   = pset.getParameter<double>("trkIsolInnR");   
    trkPtLow_           = pset.getParameter<double>("minTrackPtCut");   
    lip_                = pset.getParameter<double>("lipCut");   
    ecalIsolRadius_     = pset.getParameter<double>("ecalIsolR");    
    ecalEtaStrip_       = pset.getParameter<double>("ecalEtaStrip");

    bcEtLow_            = pset.getParameter<double>("minBcEtCut");   
    hcalIsolExtRadius_  = pset.getParameter<double>("hcalIsolExtR");   
    hcalIsolInnRadius_  = pset.getParameter<double>("hcalIsolInnR");   
    hcalHitEtLow_       = pset.getParameter<double>("minHcalHitEtCut");   

    numOfTracksInCone_  = pset.getParameter<int>("maxNumOfTracksInCone");   
    trkPtSumCut_        = pset.getParameter<double>("trkPtSumCut");   
    ecalEtSumCut_       = pset.getParameter<double>("ecalEtSumCut");   
    hcalEtSumCut_       = pset.getParameter<double>("hcalEtSumCut");   

    cutStep_            = pset.getParameter<double>("cutStep");
    numberOfSteps_      = pset.getParameter<int>("numberOfSteps");

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

  vector<string> parts;
  parts.push_back("AllEcal");
  parts.push_back("Barrel");
  parts.push_back("EndcapMinus");
  parts.push_back("EndcapPlus");

  vector<string> types;
  types.push_back("All");
  types.push_back("Isolated");
  types.push_back("Nonisolated");

  //booking all histograms

  if (dbe_) {  


    for(int cut = 0; cut != numberOfSteps_; ++cut){   //looping over Et cut values
     

      // Isolation Variable infos
     
      stringstream currentFolder;
      currentFolder << "IsolationVariables/Et above " << cut*cutStep_ << " GeV";
      dbe_->setCurrentFolder(currentFolder.str());

      h_nTrackIsol_.push_back(dbe_->book2D("nIsoTracks2D","Avg Number Of Tracks in the Iso Cone",etaBin,etaMin, etaMax,10,-0.5, 9.5));
      h_trackPtSum_.push_back(dbe_->book2D("isoPtSum2D","Avg Tracks Pt Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
      h_ecalSum_.push_back(dbe_->book2D("ecalSum2D","Avg Ecal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
      h_hcalSum_.push_back(dbe_->book2D("hcalSum2D","Avg Hcal Sum in the Iso Cone",etaBin,etaMin, etaMax,100,0., 20.));
      p_nTrackIsol_.push_back(dbe_->book1D("nIsoTracks","Avg Number Of Tracks in the Iso Cone",etaBin,etaMin, etaMax));
      p_trackPtSum_.push_back(dbe_->book1D("isoPtSum","Avg Tracks Pt Sum in the Iso Cone",etaBin,etaMin, etaMax));
      p_ecalSum_.push_back(dbe_->book1D("ecalSum","Avg Ecal Sum in the Iso Cone",etaBin,etaMin, etaMax));
      p_hcalSum_.push_back(dbe_->book1D("hcalSum","Avg Hcal Sum in the Iso Cone",etaBin,etaMin, etaMax));
   
      // Photon histograms

      for(int type=0;type!=3;++type){ //looping over isolation type
	
	currentFolder.str("");
	currentFolder << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
	dbe_->setCurrentFolder(currentFolder.str());

	for(int part=0;part!=4;++part){ //loop over different parts of the ecal

	  h_phoE_part_.push_back(dbe_->book1D("phoE"+parts[part],types[type]+" Photon Energy: "+parts[part], eBin,eMin, eMax));
	  h_phoEt_part_.push_back(dbe_->book1D("phoEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part], etBin,etMin, etMax));
	  h_r9_part_.push_back(dbe_->book1D("r9"+parts[part],types[type]+" Photon r9: "+parts[part],r9Bin,r9Min, r9Max));
	  h_nPho_part_.push_back(dbe_->book1D("nPho"+parts[part]," Number of "+types[type]+" Photons per Event: "+parts[part], 10,-0.5,9.5));
	  
	  if(part==0){
	    h_phoDistribution_part_.push_back(dbe_->book2D("Distribution"+parts[part],"Distribution of "+types[type]+" Photons in Eta/Phi: "+parts[part],phiBin,phiMin,phiMax,etaBin,-2.5,2.5));
	  }	  
	  if(part==1){
	    h_phoDistribution_part_.push_back(dbe_->book2D("Distribution"+parts[part],"Distribution of "+types[type]+" Photons in Eta/Phi: "+parts[part],360,phiMin,phiMax,170,-1.5,1.5));
	  }
	  if(part >= 2){
	    h_phoDistribution_part_.push_back(dbe_->book2D("Distribution"+parts[part],"Distribution of "+types[type]+" Photons in X/Y: "+parts[part],100,-150,150,100,-150,150));
	  }
	}

	h_phoE_isol_.push_back(h_phoE_part_);
	h_phoE_part_.clear();
	h_phoEt_isol_.push_back(h_phoEt_part_);
	h_phoEt_part_.clear();
	h_r9_isol_.push_back(h_r9_part_);
	h_r9_part_.clear();
	h_nPho_isol_.push_back(h_nPho_part_);
	h_nPho_part_.clear();
	h_phoDistribution_isol_.push_back(h_phoDistribution_part_);
	h_phoDistribution_part_.clear();

	h_phoEta_isol_.push_back(dbe_->book1D("phoEta",types[type]+" Photon Eta ",etaBin,etaMin, etaMax)) ;
	h_phoPhi_isol_.push_back(dbe_->book1D("phoPhi",types[type]+" Photon Phi ",phiBin,phiMin,phiMax)) ;
	p_r9VsEt_isol_.push_back(dbe_->bookProfile("r9VsEt",types[type]+" Photon r9 vs. Transverse Energy",etBin,etMin,etMax,r9Bin,r9Min,r9Max));

      }

    h_phoE_.push_back(h_phoE_isol_);
    h_phoE_isol_.clear();
    h_phoEt_.push_back(h_phoEt_isol_);
    h_phoEt_isol_.clear();
    h_r9_.push_back(h_r9_isol_);
    h_r9_isol_.clear();
    h_nPho_.push_back(h_nPho_isol_);
    h_nPho_isol_.clear();
    h_phoDistribution_.push_back(h_phoDistribution_isol_);
    h_phoDistribution_isol_.clear();


    h_phoEta_.push_back(h_phoEta_isol_);
    h_phoEta_isol_.clear();
    h_phoPhi_.push_back(h_phoPhi_isol_);
    h_phoPhi_isol_.clear();
    p_r9VsEt_.push_back(p_r9VsEt_isol_);
    p_r9VsEt_isol_.clear();


    }

  return ;
  
  }

}



void PhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  
  using namespace edm;






  nEvt_++;  
  LogInfo("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "PhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
 
  
  // Get the recontructed  photons
  Handle<reco::PhotonCollection> photonHandle; 
  e.getByLabel(photonProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
  std::cout  << "PhotonAnalyzer  Photons with conversions collection size " << photonCollection.size() << "\n";

  // grab PhotonId objects  
  Handle<reco::PhotonIDAssociationCollection> photonIDMapColl;
  e.getByLabel("PhotonIDProd", "PhotonAssociatedID", photonIDMapColl);
  const reco::PhotonIDAssociationCollection *phoMap = photonIDMapColl.product();

  // get the  calo topology  from the event setup:
  edm::ESHandle<CaloTopology> pTopology;
  esup.get<CaloTopologyRecord>().get(theCaloTopo_);
  const CaloTopology *topology = theCaloTopo_.product();

  // get the geometry from the event setup:
  esup.get<CaloGeometryRecord>().get(theCaloGeom_);

  // get the Hcal rec hits
  Handle<HBHERecHitCollection> hbhe;
  e.getByLabel(hbheLabel_,hbheInstanceName_,hbhe);
  if (!hbhe.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<hbheInstanceName_.c_str();
  }

  std::auto_ptr<HBHERecHitMetaCollection> mhbhe;
  mhbhe = std::auto_ptr<HBHERecHitMetaCollection>(new HBHERecHitMetaCollection(*hbhe));

  // Get the tracks
  edm::Handle<reco::TrackCollection> tracksHandle;
  e.getByLabel(tracksInputTag_,tracksHandle);
  const reco::TrackCollection* trackCollection = tracksHandle.product();







  // Creat array to hold #photons/event information
  int nPho[100][3][4];

  for (int cut=0; cut!=100; ++cut){
    for (int type=0; type!=3; ++type){
      for (int part=0; part!=4; ++part){
	nPho[cut][type][part] = 0;
      }
    }
  }


  if ( !photonHandle.isValid()) return;


  int photonCounter = 0;

  // Loop over all photons in event
  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {

    if ((*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->eta())  < minPhoEtCut_) continue;

    //edm::Ref<reco::PhotonCollection> photonref(photonCollection,1);


    edm::Ref<reco::PhotonCollection> photonref(photonHandle, photonCounter);
    photonCounter++;
    reco::PhotonIDAssociationCollection::const_iterator photonIter = phoMap->find(photonref);
    const reco::PhotonIDRef &phtn = photonIter->val;


    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;
    bool  phoIsInEndcapMinus=false;
    bool  phoIsInEndcapPlus=false;
    float etaPho=(*iPho).eta();
    if ( fabs(etaPho) <  1.479 )
      phoIsInBarrel=true;
    else {
      phoIsInEndcap=true;
      if ( etaPho < 0.0 )
	phoIsInEndcapMinus=true;
      else
	phoIsInEndcapPlus=true;
    }

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

    if (tracksHandle.isValid()) {

      PhotonTkIsolation trackerIsol(trkIsolExtRadius_, trkIsolInnRadius_, trkPtLow_, lip_, trackCollection); 
      nTracks = trackerIsol.getNumberTracks(&(*iPho));
      ptSum = trackerIsol.getPtTracks(&(*iPho));

    }

    /// isolation in Ecal
    edm::Handle<EcalRecHitCollection> ecalRecHitHandle;
    edm::Handle<reco::BasicClusterCollection> bcHandle;
    edm::Handle<reco::SuperClusterCollection> scHandle;


    if ( scIsInBarrel ) {     

      // Get the Ecal barrel Rec hits
      e.getByLabel(barrelEcalHits_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
      }
      
      // Get the basic cluster collection in the Barrel 
      e.getByLabel(bcBarrelProducer_, bcBarrelCollection_, bcHandle);
      if (!bcHandle.isValid()) {
	edm::LogError("ConversionTrackCandidateProducer") << "Error! Can't get the product "<<bcBarrelCollection_.c_str();
      }

      // Get the Super Cluster collection in the Barrel
      e.getByLabel(scBarrelProducer_,scHandle);
      if (!scHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<scBarrelProducer_.label();
      }
  

    } else if ( scIsInEndcap ) {    

      // Get the Ecal endcap Rec hits
      e.getByLabel(endcapEcalHits_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
      }
     
      // Get the basic cluster collection in the Endcap 
      e.getByLabel(bcEndcapProducer_, bcEndcapCollection_, bcHandle);
      if (!bcHandle.isValid()) {
	edm::LogError("ConversionTrackCandidateProducer") << "Error! Can't get the product "<<bcEndcapCollection_.c_str();
      }

      // Get the Super Cluster collection in the Endcap
      e.getByLabel(scEndcapProducer_,scHandle);
      if (!scHandle.isValid()) {
	edm::LogError("PhotonProducer") << "Error! Can't get the product "<<scEndcapProducer_.label();
      }


    }

    const EcalRecHitCollection ecalRecHitCollection = *(ecalRecHitHandle.product());
    const reco::SuperClusterCollection scCollection = *(scHandle.product());
    const reco::BasicClusterCollection bcCollection = *(bcHandle.product());

    /// isolation in Ecal
    if ( bcHandle.isValid() && scHandle.isValid() ) {

    EgammaEcalIsolation ecalIsol( ecalIsolRadius_, bcEtLow_, &bcCollection, &scCollection);
    ecalSum = ecalIsol.getEcalEtSum(&(*iPho));  

    }

    /// isolation in Hcal
    if ( hbhe.isValid() ) {

    EgammaHcalIsolation hcalIsol (hcalIsolExtRadius_,hcalIsolInnRadius_,hcalHitEtLow_,theCaloGeom_.product(),mhbhe.get()); 
    hcalSum = hcalIsol.getHcalEtSum(&(*iPho)); 

    }
    
    bool isIsolated=false;

    //old version

//     if ( (nTracks < numOfTracksInCone_) && 
// 	     ( ptSum < trkPtSumCut_) &&
// 	     ( ecalSum < ecalEtSumCut_ ) &&
// 	     ( hcalSum < hcalEtSumCut_ ) ) isIsolated = true;

    //new version for 2_1_4 and up

    //isIsolated = (phtn)->isTightPhoton();
    isIsolated = (phtn)->isLoosePhoton();

    int type=0;
    if ( isIsolated ) type=1;
    if ( !isIsolated ) type=2;

    

    nEntry_++;

    float e3x3=   EcalClusterTools::e3x3(  *(   (*iPho).superCluster()->seed()  ), &ecalRecHitCollection, &(*topology)); 
    float r9 =e3x3/( (*iPho).superCluster()->rawEnergy()+ (*iPho).superCluster()->preshowerEnergy());

    for (int cut=0; cut !=numberOfSteps_; ++cut) {
      double Et =  (*iPho).energy()/cosh( (*iPho).superCluster()->eta());
      
      if ( Et > cut*cutStep_ && ( Et < (cut+1)*cutStep_  | cut == numberOfSteps_-1 ) ){
      //if ( Et > cut*cutStep_ ){

	//filling isolation variable histograms

	//old version

// 	h_nTrackIsol_[cut]->Fill( (*iPho).superCluster()->eta(), float(nTracks));
// 	h_trackPtSum_[cut]->Fill((*iPho).superCluster()->eta(), ptSum);
	
// 	h_ecalSum_[cut]->Fill((*iPho).superCluster()->eta(), ecalSum);
// 	h_hcalSum_[cut]->Fill((*iPho).superCluster()->eta(), hcalSum);
  
	//new version for 2_1_4 and up
	
	h_nTrackIsol_[cut]->Fill( (*iPho).superCluster()->eta(),(phtn)->nTrkSolidCone());
	h_trackPtSum_[cut]->Fill((*iPho).superCluster()->eta(), (phtn)->isolationSolidTrkCone());
	
	h_ecalSum_[cut]->Fill((*iPho).superCluster()->eta(), (phtn)->isolationEcalRecHit());
	h_hcalSum_[cut]->Fill((*iPho).superCluster()->eta(), (phtn)->isolationHcalRecHit());


	//filling all photons histograms
	h_phoE_[cut][0][0]->Fill( (*iPho).energy() );
	h_phoEt_[cut][0][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );

	h_r9_[cut][0][0]->Fill( r9 );
	h_phoDistribution_[cut][0][0]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
	nPho[cut][0][0]++;

	h_phoEta_[cut][0]->Fill( (*iPho).superCluster()->eta() );
	h_phoPhi_[cut][0]->Fill( (*iPho).superCluster()->phi() );      
    
	p_r9VsEt_[cut][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()), r9 );

	// iso/noniso photons histograms

	h_phoE_[cut][type][0]->Fill( (*iPho).energy() );
	h_phoEt_[cut][type][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	nPho[cut][type][0]++;
	h_phoDistribution_[cut][type][0]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
	h_r9_[cut][type][0]->Fill( r9 );

	h_phoEta_[cut][type]->Fill( (*iPho).superCluster()->eta() );
	h_phoPhi_[cut][type]->Fill( (*iPho).superCluster()->phi() );      

	p_r9VsEt_[cut][type]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()), r9 );

	//filling both types of histograms

	if ( phoIsInBarrel ) { 
	  h_phoE_[cut][0][1]->Fill( (*iPho).energy() );
	  h_phoEt_[cut][0][1]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	  nPho[cut][0][1]++;
	  h_phoDistribution_[cut][0][1]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
	  h_r9_[cut][0][1]->Fill( r9 );
	  
	  h_phoE_[cut][type][1]->Fill( (*iPho).energy() );
	  h_phoEt_[cut][type][1]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	  nPho[cut][type][1]++;
	  h_phoDistribution_[cut][type][1]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
	  h_r9_[cut][type][1]->Fill( r9 );
	}	  

	int part = 0;
	if ( phoIsInEndcap ) {
	  if (phoIsInEndcapMinus)
	    part = 2;
	  else if (phoIsInEndcapPlus)
	    part = 3;

	  h_phoE_[cut][0][part]->Fill( (*iPho).energy() );
	  h_phoEt_[cut][0][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	  nPho[cut][0][part]++;
	  h_phoDistribution_[cut][0][part]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	  h_r9_[cut][0][part]->Fill( r9 );

	  h_phoE_[cut][type][part]->Fill( (*iPho).energy() );
	  h_phoEt_[cut][type][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );
	  nPho[cut][type][part]++;
	  h_phoDistribution_[cut][type][part]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );
	  h_r9_[cut][type][part]->Fill( r9 );

	}
      
      }

    }



  }/// End loop over Reco  particles
    
  //filling number of photons per event histograms
  for (int cut=0; cut !=numberOfSteps_; ++cut) {
    for(int type=0;type!=3;++type){
      for(int part=0;part!=4;++part){
	h_nPho_[cut][type][part]-> Fill (float(nPho[cut][type][part]));
      }
    }
  }

}


void PhotonAnalyzer::endJob()
{

  for (int cut=0; cut !=numberOfSteps_; ++cut) {

     doProfileX( h_nTrackIsol_[cut], p_nTrackIsol_[cut]);
     doProfileX( h_trackPtSum_[cut], p_trackPtSum_[cut]);
     doProfileX( h_ecalSum_[cut], p_ecalSum_[cut]);
     doProfileX( h_hcalSum_[cut], p_hcalSum_[cut]);
  
  }


  bool outputMEsInRootFile = parameters_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
  
  edm::LogInfo("PhotonAnalyzer") << "Analyzed " << nEvt_  << "\n";
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

 return phi;

}


void PhotonAnalyzer::doProfileX(TH2 * th2, MonitorElement* me){
  if (th2->GetNbinsX()==me->getNbinsX()){
    TH1F * h1 = (TH1F*) th2->ProfileX();
    for (int bin=0;bin!=h1->GetNbinsX();bin++){
      me->setBinContent(bin+1,h1->GetBinContent(bin+1));
      me->setBinError(bin+1,h1->GetBinError(bin+1));
    }
    delete h1;
  } else {
    throw cms::Exception("PhotonAnalyzer") << "Different number of bins!";
  }
}

void PhotonAnalyzer::doProfileX(MonitorElement * th2m, MonitorElement* me) {
  doProfileX(th2m->getTH2F(), me);
}

