#include "RecoParticleFlow/PFProducer/plugins/PFBlockProducer.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"


#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <set>

using namespace std;
using namespace edm;

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig) {

  // use configuration file to setup input/output collection names
  inputTagRecTracks_ =consumes<reco::PFRecTrackCollection>(iConfig.getParameter<InputTag>("RecTracks"));


  inputTagGsfRecTracks_ =consumes<reco::GsfPFRecTrackCollection>(iConfig.getParameter<InputTag>("GsfRecTracks"));

  inputTagConvBremGsfRecTracks_ =consumes<reco::GsfPFRecTrackCollection>(iConfig.getParameter<InputTag>("ConvBremGsfRecTracks"));

  inputTagRecMuons_ =consumes<reco::MuonCollection>(iConfig.getParameter<InputTag>("RecMuons"));

  inputTagPFNuclear_ =consumes<reco::PFDisplacedTrackerVertexCollection>(iConfig.getParameter<InputTag>("PFNuclear"));

  inputTagPFConversions_ =consumes<reco::PFConversionCollection>(iConfig.getParameter<InputTag>("PFConversions"));

  inputTagPFV0_ =consumes<reco::PFV0Collection>(iConfig.getParameter<InputTag>("PFV0"));

  inputTagPFClustersECAL_ =consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFClustersECAL"));

  inputTagPFClustersHCAL_ =consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFClustersHCAL"));

  inputTagPFClustersHO_ =consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFClustersHO"));

  inputTagPFClustersHFEM_ =consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFClustersHFEM"));

  inputTagPFClustersHFHAD_ =consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFClustersHFHAD"));

  inputTagPFClustersPS_ =consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFClustersPS"));

  useEGPhotons_ = iConfig.getParameter<bool>("useEGPhotons");
  
  if(useEGPhotons_) {
    inputTagEGPhotons_=consumes<reco::PhotonCollection>(iConfig.getParameter<InputTag>("EGPhotons"));         
  }
  
  useSuperClusters_ = iConfig.existsAs<bool>("useSuperClusters") ? iConfig.getParameter<bool>("useSuperClusters") : false;
  
  if (useSuperClusters_) {
    inputTagSCBarrel_=consumes<reco::SuperClusterCollection>(iConfig.getParameter<InputTag>("SCBarrel"));      
    inputTagSCEndcap_=consumes<reco::SuperClusterCollection>(iConfig.getParameter<InputTag>("SCEndcap"));     
  }
  
  //default value = false (for compatibility with old HLT configs)
  superClusterMatchByRef_ = iConfig.existsAs<bool>("SuperClusterMatchByRef") ? iConfig.getParameter<bool>("SuperClusterMatchByRef") : false;
  
  if (superClusterMatchByRef_) {
    inputTagPFClusterAssociationEBEE_ =consumes<edm::ValueMap<reco::CaloClusterPtr> >(iConfig.getParameter<InputTag>("PFClusterAssociationEBEE"));
  }
  
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  bool debug_ = 
    iConfig.getUntrackedParameter<bool>("debug",false);

  usePFatHLT_ = iConfig.getParameter<bool>("usePFatHLT");

  useNuclear_ = iConfig.getParameter<bool>("useNuclear");

  useConversions_ = iConfig.getParameter<bool>("useConversions");
  
  useConvBremGsfTracks_ = iConfig.getParameter<bool>("useConvBremGsfTracks");

  bool useConvBremPFRecTracks = iConfig.getParameter<bool>("useConvBremPFRecTracks");

  useV0_ = iConfig.getParameter<bool>("useV0");

  useHO_=  iConfig.getParameter<bool>("useHO");

  produces<reco::PFBlockCollection>();
  
  // Glowinski & Gouzevitch
  useKDTreeTrackEcalLinker_ = iConfig.getParameter<bool>("useKDTreeTrackEcalLinker");
  // !Glowinski & Gouzevitch
  
  // particle flow parameters  -----------------------------------

  std::vector<double> DPtovPtCut 
     = iConfig.getParameter<std::vector<double> >("pf_DPtoverPt_Cut");   
  if (DPtovPtCut.size()!=5)
    {
      edm::LogError("MisConfiguration")<<" vector pf_DPtoverPt_Cut has to be of size 5";
      throw;
    }

  std::vector<unsigned> NHitCut 
     = iConfig.getParameter<std::vector<unsigned> >("pf_NHit_Cut");   
  if (NHitCut.size()!=5)
    {
      edm::LogError("MisConfiguration")<<" vector pf_NHit_Cut has to be of size 5";
      throw;
    }

  bool useIterTracking
    = iConfig.getParameter<bool>("useIterTracking");

  int nuclearInteractionsPurity
    = iConfig.getParameter<unsigned>("nuclearInteractionsPurity");

  // if first parameter 0, deactivated 
  std::vector<double> EGPhotonSelectionCuts ;

  if (useEGPhotons_)
    EGPhotonSelectionCuts = iConfig.getParameter<std::vector<double> >("PhotonSelectionCuts");   


  if (useNuclear_){
    if (nuclearInteractionsPurity > 3 || nuclearInteractionsPurity < 1)  {
      nuclearInteractionsPurity = 1;
      edm::LogInfo("PFBlockProducer")  << "NI purity not properly implemented. Set it to the strongest level " << nuclearInteractionsPurity << endl;
    }
    vector<string> securityLevel;
    securityLevel.push_back("isNucl"); securityLevel.push_back("isNucl && isNuclLoose");  securityLevel.push_back("isNucl && isNuclLoose && isNuclKink"); 
    edm::LogInfo("PFBlockProducer")  << "NI interactions are corrected in PFlow for " << securityLevel[nuclearInteractionsPurity-1].c_str() << endl;
  }


  pfBlockAlgo_.setParameters( DPtovPtCut,
			      NHitCut,
			      useConvBremPFRecTracks,
			      useIterTracking,
			      nuclearInteractionsPurity,
			      useEGPhotons_,
			      EGPhotonSelectionCuts,
			      useSuperClusters_,
                              superClusterMatchByRef_
			    );
  
  pfBlockAlgo_.setDebug(debug_);

  // Glowinski & Gouzevitch
  pfBlockAlgo_.setUseOptimization(useKDTreeTrackEcalLinker_);
  // !Glowinski & Gouzevitch

  // Use HO clusters for link
  pfBlockAlgo_.setHOTag(useHO_);

}



PFBlockProducer::~PFBlockProducer() { }



void 
PFBlockProducer::produce(Event& iEvent, 
			 const EventSetup& iSetup) {
  
  // get rectracks
  
  Handle< reco::PFRecTrackCollection > recTracks;
  
  // LogDebug("PFBlockProducer")<<"get reco tracks"<<endl;
   iEvent.getByToken(inputTagRecTracks_, recTracks);
 

  // get GsfTracks 
  Handle< reco::GsfPFRecTrackCollection > GsfrecTracks;

  if(!usePFatHLT_) 
    iEvent.getByToken(inputTagGsfRecTracks_,GsfrecTracks);

  // get ConvBremGsfTracks 
  Handle< reco::GsfPFRecTrackCollection > convBremGsfrecTracks;

  if(useConvBremGsfTracks_) 
    iEvent.getByToken(inputTagConvBremGsfRecTracks_,convBremGsfrecTracks);


  // get recmuons
  Handle< reco::MuonCollection > recMuons;

  iEvent.getByToken(inputTagRecMuons_, recMuons);
  Handle< reco::PFDisplacedTrackerVertexCollection > pfNuclears; 

  if( useNuclear_ ) {
    iEvent.getByToken(inputTagPFNuclear_, pfNuclears);
  }
 
  // get conversions
  Handle< reco::PFConversionCollection > pfConversions;
  if( useConversions_ ) {
    iEvent.getByToken(inputTagPFConversions_, pfConversions);
  }

  // get V0s
  Handle< reco::PFV0Collection > pfV0;
  if( useV0_ ) {
    iEvent.getByToken(inputTagPFV0_, pfV0);
  }


  
  // get ECAL, HCAL, HO and PS clusters
  
  
  Handle< reco::PFClusterCollection > clustersECAL;
  iEvent.getByToken(inputTagPFClustersECAL_, 
			    clustersECAL);      
    
  Handle< reco::PFClusterCollection > clustersHCAL;
  iEvent.getByToken(inputTagPFClustersHCAL_, 
			    clustersHCAL);      
      
  Handle< reco::PFClusterCollection > clustersHO;
  if (useHO_) {
    iEvent.getByToken(inputTagPFClustersHO_, 
			      clustersHO);      
  }

  Handle< reco::PFClusterCollection > clustersHFEM;
  iEvent.getByToken(inputTagPFClustersHFEM_, 
			    clustersHFEM);      
      
  Handle< reco::PFClusterCollection > clustersHFHAD;
  iEvent.getByToken(inputTagPFClustersHFHAD_, 
			    clustersHFHAD);      

  Handle< reco::PFClusterCollection > clustersPS;
  iEvent.getByToken(inputTagPFClustersPS_, 
			    clustersPS);      
   // dummy. Not used in the full framework 
  Handle< reco::PFRecTrackCollection > nuclearRecTracks;
  
  Handle< reco::PhotonCollection >  egPhotons;
  if (useEGPhotons_)
    iEvent.getByToken(inputTagEGPhotons_,
			    egPhotons);
  			       
  Handle< reco::SuperClusterCollection >  sceb;
  Handle< reco::SuperClusterCollection >  scee;
  
  if (useSuperClusters_) {
    iEvent.getByToken(inputTagSCBarrel_,
			      sceb);
    iEvent.getByToken(inputTagSCEndcap_,
			      scee);
  }
  
  Handle<edm::ValueMap<reco::CaloClusterPtr> > pfclusterassoc;
  if (superClusterMatchByRef_) {
    iEvent.getByToken(inputTagPFClusterAssociationEBEE_,
                              pfclusterassoc);
  }

  if( usePFatHLT_  ) {
     pfBlockAlgo_.setInput( recTracks, 		
			    recMuons,
			    clustersECAL,
			    clustersHCAL,
			    clustersHO,
			    clustersHFEM,
			    clustersHFHAD,
			    clustersPS);
  } else { 
    pfBlockAlgo_.setInput( recTracks, 
			   GsfrecTracks,
			   convBremGsfrecTracks,
			   recMuons, 
			   pfNuclears,
			   nuclearRecTracks,
			   pfConversions,
			   pfV0,
			   clustersECAL,
			   clustersHCAL,
			   clustersHO,
			   clustersHFEM,
			   clustersHFHAD,
			   clustersPS,
			   egPhotons,
			   sceb,
			   scee,
                           pfclusterassoc
                         );
  }
  pfBlockAlgo_.findBlocks();
  
  if(verbose_) {
    ostringstream  str;
    str<<pfBlockAlgo_<<endl;
    LogInfo("PFBlockProducer") << str.str()<<endl;
  }    

  auto_ptr< reco::PFBlockCollection > 
    pOutputBlockCollection( pfBlockAlgo_.transferBlocks() ); 
  
  iEvent.put(pOutputBlockCollection);
    
}
