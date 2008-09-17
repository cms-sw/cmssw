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

/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
 **  $Date:  $ 
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

    
    photonProducer_     = pset.getParameter<std::string>("phoProducer");
    photonCollection_   = pset.getParameter<std::string>("photonCollection");

    minPhoEtCut_        = pset.getParameter<double>("minPhoEtCut");   

    cutStep_            = pset.getParameter<double>("cutStep");
    numberOfSteps_      = pset.getParameter<int>("numberOfSteps");

    useBinning_         = pset.getParameter<bool>("useBinning");

    isolationStrength_  = pset.getParameter<int>("isolationStrength");

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
  int dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");

 

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
   

      // Photon histograms

      for(int type=0;type!=3;++type){ //looping over isolation type
	
	currentFolder.str("");
	currentFolder << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV";
	dbe_->setCurrentFolder(currentFolder.str());

	for(int part=0;part!=4;++part){ //loop over different parts of the ecal

	  h_phoE_part_.push_back(dbe_->book1D("phoE"+parts[part],types[type]+" Photon Energy: "+parts[part], eBin,eMin, eMax));
	  h_phoEt_part_.push_back(dbe_->book1D("phoEt"+parts[part],types[type]+" Photon Transverse Energy: "+parts[part], etBin,etMin, etMax));
	  h_r9_part_.push_back(dbe_->book1D("r9"+parts[part],types[type]+" Photon r9: "+parts[part],r9Bin,r9Min, r9Max));
	  h_hOverE_part_.push_back(dbe_->book1D("hOverE"+parts[part],types[type]+" Photon H/E: "+parts[part],r9Bin,r9Min, 1));
	  h_nPho_part_.push_back(dbe_->book1D("nPho"+parts[part],"Number of "+types[type]+" Photons per Event: "+parts[part], 10,-0.5,9.5));

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
	h_hOverE_isol_.push_back(h_hOverE_part_);
	h_hOverE_part_.clear();
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
      p_r9VsEt_.push_back(p_r9VsEt_isol_);
      p_r9VsEt_isol_.clear();
      
   
    
      //conversion plots

      for(int type=0;type!=3;++type){ //looping over isolation type

	stringstream currentFolder;
	currentFolder << types[type] << "Photons/Et above " << cut*cutStep_ << " GeV/Conversions";
	dbe_->setCurrentFolder(currentFolder.str());

	for(int part=0;part!=4;++part){ //loop over different parts of the ecal

	  h_nConv_part_.push_back(dbe_->book1D("nConv"+parts[part],"Number Of Conversions per Event:  "+parts[part] ,10,-0.5, 9.5));
	  h_eOverPTracks_part_.push_back(dbe_->book1D("eOverPTracks"+parts[part],"E/P of Conversions: "+parts[part] ,100, 0., 5.));
	  h_dPhiTracksAtVtx_part_.push_back(dbe_->book1D("dPhiTracksAtVtx"+parts[part], "  #delta#phi of Conversion Tracks at Vertex: "+parts[part],dPhiTracksBin,dPhiTracksMin,dPhiTracksMax)); 
	  h_dCotTracks_part_.push_back(dbe_->book1D("dCotTracks"+parts[part],"#delta cotg(#Theta) of Conversion Tracks: "+parts[part],dEtaTracksBin,dEtaTracksMin,dEtaTracksMax)); 

	}

	h_nConv_isol_.push_back(h_nConv_part_);
	h_nConv_part_.clear();
	h_eOverPTracks_isol_.push_back(h_eOverPTracks_part_);
	h_eOverPTracks_part_.clear();
	h_dPhiTracksAtVtx_isol_.push_back(h_dPhiTracksAtVtx_part_);
	h_dPhiTracksAtVtx_part_.clear();
	h_dCotTracks_isol_.push_back(h_dCotTracks_part_);
	h_dCotTracks_part_.clear();

	h_phoConvEta_isol_.push_back(dbe_->book1D("phoConvEta",types[type]+" Converted Photon Eta ",etaBin,etaMin, etaMax)) ;
	h_phoConvPhi_isol_.push_back(dbe_->book1D("phoConvPhi",types[type]+" Converted Photon Phi ",phiBin,phiMin,phiMax)) ;
	h_convVtxRvsZ_isol_.push_back(dbe_->book2D("convVtxRvsZ",types[type]+" Photon Reco conversion vtx position",100, 0., 280.,200,0., 120.));

      }

      h_nConv_.push_back(h_nConv_isol_);
      h_nConv_isol_.clear();
      h_eOverPTracks_.push_back(h_eOverPTracks_isol_);
      h_eOverPTracks_isol_.clear();
      h_dPhiTracksAtVtx_.push_back(h_dPhiTracksAtVtx_isol_);
      h_dPhiTracksAtVtx_isol_.clear();
      h_dCotTracks_.push_back(h_dCotTracks_isol_);
      h_dCotTracks_isol_.clear();
      
      h_phoConvEta_.push_back(h_phoConvEta_isol_);
      h_phoConvEta_isol_.clear();
      h_phoConvPhi_.push_back(h_phoConvPhi_isol_);
      h_phoConvPhi_isol_.clear();
      h_convVtxRvsZ_.push_back(h_convVtxRvsZ_isol_);
      h_convVtxRvsZ_isol_.clear();
  
    }

    return ;
    
  }
  
}



void PhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  
  using namespace edm;

  nEvt_++;  
  LogInfo("PhotonAnalyzer") << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
 
  
  // Get the recontructed  photons
  Handle<reco::PhotonCollection> photonHandle; 
  e.getByLabel(photonProducer_, photonCollection_ , photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
 

  // grab PhotonId objects  
  Handle<reco::PhotonIDAssociationCollection> photonIDMapColl;
  e.getByLabel("PhotonIDProd", "PhotonAssociatedID", photonIDMapColl);
  const reco::PhotonIDAssociationCollection *phoMap = photonIDMapColl.product();


  // get the geometry from the event setup:
  //esup.get<CaloGeometryRecord>().get(theCaloGeom_);

  // Create array to hold #photons/event information
  int nPho[100][3][4];

  for (int cut=0; cut!=100; ++cut){
    for (int type=0; type!=3; ++type){
      for (int part=0; part!=4; ++part){
	nPho[cut][type][part] = 0;
      }
    }
  }


  if ( !photonHandle.isValid()) return;
  if ( !photonIDMapColl.isValid()) return;

  int photonCounter = 0;

  // Loop over all photons in event
  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {

    if ((*iPho).superCluster()->energy()/cosh( (*iPho).superCluster()->eta())  < minPhoEtCut_) continue;

    edm::Ref<reco::PhotonCollection> photonref(photonHandle, photonCounter);
    photonCounter++;
    reco::PhotonIDAssociationCollection::const_iterator photonIter = phoMap->find(photonref);
    const reco::PhotonIDRef &phtn = photonIter->val;


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


    bool isIsolated=false;
    if ( isolationStrength_ == 1)  isIsolated = (phtn)->isLooseEM();
    else if ( isolationStrength_ == 2)  isIsolated = (phtn)->isLoosePhoton(); 
    else if ( isolationStrength_ == 3)  isIsolated = (phtn)->isTightPhoton();
 
    int type=0;
    if ( isIsolated ) type=1;
    if ( !isIsolated ) type=2;

    
    nEntry_++;

    float r9 = (phtn)->r9();

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
	h_nTrackIsolSolid_[cut]->Fill( (*iPho).superCluster()->eta(),(phtn)->nTrkSolidCone());
	h_trackPtSumSolid_[cut]->Fill((*iPho).superCluster()->eta(), (phtn)->isolationSolidTrkCone());
	
	h_nTrackIsolHollow_[cut]->Fill( (*iPho).superCluster()->eta(),(phtn)->nTrkHollowCone());
	h_trackPtSumHollow_[cut]->Fill((*iPho).superCluster()->eta(), (phtn)->isolationHollowTrkCone());

	h_ecalSum_[cut]->Fill((*iPho).superCluster()->eta(), (phtn)->isolationEcalRecHit());
	h_hcalSum_[cut]->Fill((*iPho).superCluster()->eta(), (phtn)->isolationHcalRecHit());


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
    
	p_r9VsEt_[cut][0]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()), r9 );


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

	p_r9VsEt_[cut][type]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()), r9 );


	//filling both types of histograms for different ecal parts
	int part = 0;
	if ( phoIsInBarrel )
	  part = 1;
	if ( phoIsInEndcap ) {
	  if (phoIsInEndcapMinus)
	    part = 2;
	  else if (phoIsInEndcapPlus)
	    part = 3;
	}

	h_phoE_[cut][0][part]->Fill( (*iPho).energy() );
	h_phoEt_[cut][0][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );

	h_r9_[cut][0][part]->Fill( r9 );
	h_hOverE_[cut][0][part]->Fill( (*iPho).hadronicOverEm() );	

	nPho[cut][0][part]++;
	h_nConv_[cut][0][part]->Fill(float( (*iPho).conversions().size() ));

	if (part == 1)  h_phoDistribution_[cut][0][part]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
	else  h_phoDistribution_[cut][0][part]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );


	h_phoE_[cut][type][part]->Fill( (*iPho).energy() );
	h_phoEt_[cut][type][part]->Fill( (*iPho).energy()/ cosh( (*iPho).superCluster()->eta()) );

	h_r9_[cut][type][part]->Fill( r9 );
	h_hOverE_[cut][type][part]->Fill( (*iPho).hadronicOverEm() );

	nPho[cut][type][part]++;
	h_nConv_[cut][type][part]->Fill(float( (*iPho).conversions().size() ));

	if (part == 1)  h_phoDistribution_[cut][type][part]->Fill( (*iPho).superCluster()->phi(),(*iPho).superCluster()->eta() );
	else  h_phoDistribution_[cut][type][part]->Fill( (*iPho).superCluster()->x(),(*iPho).superCluster()->y() );	

	//loop over conversions

	std::vector<reco::ConversionRef> conversions = (*iPho).conversions();
	for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

	  h_phoConvEta_[cut][0]->Fill( conversions[iConv]->caloCluster()[0]->eta()  );
	  h_phoConvPhi_[cut][0]->Fill( conversions[iConv]->caloCluster()[0]->phi()  );  
	  h_phoConvEta_[cut][type]->Fill( conversions[iConv]->caloCluster()[0]->eta()  );
	  h_phoConvPhi_[cut][type]->Fill( conversions[iConv]->caloCluster()[0]->phi()  );  


	  if ( conversions[iConv]->conversionVertex().isValid() ) {

	    h_convVtxRvsZ_[cut][0] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
					       sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	    h_convVtxRvsZ_[cut][type] ->Fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  
					       sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
	  }


	  std::vector<reco::TrackRef> tracks = conversions[iConv]->tracks();
	  float  DPhiTracksAtVtx = -99;

	  if ( tracks.size() > 1 ) {
	    float phiTk1= tracks[0]->innerMomentum().phi();
	    float phiTk2= tracks[1]->innerMomentum().phi();
	    DPhiTracksAtVtx = phiTk1-phiTk2;
	    DPhiTracksAtVtx = phiNormalization( DPhiTracksAtVtx );
	  }


	  h_eOverPTracks_[cut][0][0] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_eOverPTracks_[cut][type][0] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_dCotTracks_[cut][0][0] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dCotTracks_[cut][type][0] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dPhiTracksAtVtx_[cut][0][0]->Fill( DPhiTracksAtVtx);
	  h_dPhiTracksAtVtx_[cut][type][0]->Fill( DPhiTracksAtVtx);


	  //filling both types of histograms for different ecal parts

	  int part = 0;
	  if ( phoIsInBarrel )
	    part = 1;
	  if ( phoIsInEndcap ) {
	    if (phoIsInEndcapMinus)
	      part = 2;
	    else if (phoIsInEndcapPlus)
	      part = 3;
	  }

	  
	  h_eOverPTracks_[cut][0][part] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_eOverPTracks_[cut][type][part] ->Fill( conversions[iConv]->EoverP() ) ;
	  h_dCotTracks_[cut][0][part] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dCotTracks_[cut][type][part] ->Fill ( conversions[iConv]->pairCotThetaSeparation() );	  
	  h_dPhiTracksAtVtx_[cut][0][part]->Fill( DPhiTracksAtVtx);
	  h_dPhiTracksAtVtx_[cut][type][part]->Fill( DPhiTracksAtVtx);


	}//end loop over conversions

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

     doProfileX( h_nTrackIsolSolid_[cut], p_nTrackIsolSolid_[cut]);
     doProfileX( h_trackPtSumSolid_[cut], p_trackPtSumSolid_[cut]);
     doProfileX( h_nTrackIsolHollow_[cut], p_nTrackIsolHollow_[cut]);
     doProfileX( h_trackPtSumHollow_[cut], p_trackPtSumHollow_[cut]);
     doProfileX( h_ecalSum_[cut], p_ecalSum_[cut]);
     doProfileX( h_hcalSum_[cut], p_hcalSum_[cut]);

     stringstream currentFolder;
     currentFolder << "IsolationVariables/Et above " << cut*cutStep_ << " GeV";
     dbe_->setCurrentFolder(currentFolder.str());

     dbe_->removeElement("nIsoTracksSolid2D");
     dbe_->removeElement("nIsoTracksHollow2D");
     dbe_->removeElement("isoPtSumSolid2D");
     dbe_->removeElement("isoPtSumHollow2D");
     dbe_->removeElement("ecalSum2D");
     dbe_->removeElement("hcalSum2D");


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
    delete h1;
  } else {
    throw cms::Exception("PhotonAnalyzer") << "Different number of bins!";
  }
}

void PhotonAnalyzer::doProfileX(MonitorElement * th2m, MonitorElement* me) {
  doProfileX(th2m->getTH2F(), me);
}

