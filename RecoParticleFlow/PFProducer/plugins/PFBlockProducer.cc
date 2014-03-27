#include "RecoParticleFlow/PFProducer/plugins/PFBlockProducer.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h" 
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

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
  inputTagRecTracks_ 
    = iConfig.getParameter<InputTag>("RecTracks");

  inputTagGsfRecTracks_ 
    = iConfig.getParameter<InputTag>("GsfRecTracks");

    inputTagConvBremGsfRecTracks_ 
    = iConfig.getParameter<InputTag>("ConvBremGsfRecTracks");

  inputTagRecMuons_ 
    = iConfig.getParameter<InputTag>("RecMuons");

  inputTagPFNuclear_ 
    = iConfig.getParameter<InputTag>("PFNuclear");

  inputTagPFConversions_ 
    = iConfig.getParameter<InputTag>("PFConversions");

  inputTagPFV0_ 
    = iConfig.getParameter<InputTag>("PFV0");

  inputTagPFClustersECAL_ 
    = iConfig.getParameter<InputTag>("PFClustersECAL");

  inputTagPFClustersHCAL_ 
    = iConfig.getParameter<InputTag>("PFClustersHCAL");

  inputTagPFClustersHO_ 
    = iConfig.getParameter<InputTag>("PFClustersHO");

  inputTagPFClustersHFEM_ 
    = iConfig.getParameter<InputTag>("PFClustersHFEM");

  inputTagPFClustersHFHAD_ 
    = iConfig.getParameter<InputTag>("PFClustersHFHAD");

  inputTagPFClustersPS_ 
    = iConfig.getParameter<InputTag>("PFClustersPS");

  useEGPhotons_ = iConfig.getParameter<bool>("useEGPhotons");
  
  if(useEGPhotons_) {
    inputTagEGPhotons_
      = iConfig.getParameter<InputTag>("EGPhotons");         
  }
  
  useSuperClusters_ = iConfig.existsAs<bool>("useSuperClusters") ? iConfig.getParameter<bool>("useSuperClusters") : false;
  
  if (useSuperClusters_) {
    inputTagSCBarrel_
      = iConfig.getParameter<InputTag>("SCBarrel");      
    inputTagSCEndcap_
      = iConfig.getParameter<InputTag>("SCEndcap");     
  }
  
  //default value = false (for compatibility with old HLT configs)
  superClusterMatchByRef_ = iConfig.existsAs<bool>("SuperClusterMatchByRef") ? iConfig.getParameter<bool>("SuperClusterMatchByRef") : false;
  
  if (superClusterMatchByRef_) {
    inputTagPFClusterAssociationEBEE_ = iConfig.getParameter<InputTag>("PFClusterAssociationEBEE");
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
  
  LogDebug("PFBlockProducer")<<"START event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;
  
  
  // get rectracks
  
  Handle< reco::PFRecTrackCollection > recTracks;
  
  // LogDebug("PFBlockProducer")<<"get reco tracks"<<endl;
  bool found = iEvent.getByLabel(inputTagRecTracks_, recTracks);
    
  if(!found )
    LogError("PFBlockProducer")<<" cannot get rectracks: "
			       <<inputTagRecTracks_<<endl;



  // get GsfTracks 
  Handle< reco::GsfPFRecTrackCollection > GsfrecTracks;

  if(!usePFatHLT_) {
    found = iEvent.getByLabel(inputTagGsfRecTracks_,GsfrecTracks);

    if(!found )
      LogError("PFBlockProducer")<<" cannot get Gsfrectracks: "
				 << inputTagGsfRecTracks_ <<endl;
  }

  // get ConvBremGsfTracks 
  Handle< reco::GsfPFRecTrackCollection > convBremGsfrecTracks;

  if(useConvBremGsfTracks_) {
    found = iEvent.getByLabel(inputTagConvBremGsfRecTracks_,convBremGsfrecTracks);

    if(!found )
      LogError("PFBlockProducer")<<" cannot get ConvBremGsfrectracks: "
				 << inputTagConvBremGsfRecTracks_ <<endl;
  }

  // get recmuons
  Handle< reco::MuonCollection > recMuons;

  // LogDebug("PFBlockProducer")<<"get reco muons"<<endl;
  //if(!usePFatHLT_) {
  found = iEvent.getByLabel(inputTagRecMuons_, recMuons);
  
    //if(!found )
    //  LogError("PFBlockProducer")<<" cannot get recmuons: "
    //			 <<inputTagRecMuons_<<endl;

  // get PFNuclearInteractions
  //}
  //---------- Gouzevitch
  //  Handle< reco::PFNuclearInteractionCollection > pfNuclears; 
  Handle< reco::PFDisplacedTrackerVertexCollection > pfNuclears; 

  if( useNuclear_ ) {

    found = iEvent.getByLabel(inputTagPFNuclear_, pfNuclears);
    if(!found )
      LogError("PFBlockProducer")<<" cannot get PFNuclearInteractions : "
                               <<inputTagPFNuclear_<<endl;
  }
 
  // get conversions
  Handle< reco::PFConversionCollection > pfConversions;
  if( useConversions_ ) {
    found = iEvent.getByLabel(inputTagPFConversions_, pfConversions);
    
    if(!found )
      LogError("PFBlockProducer")<<" cannot get PFConversions : "
				 <<inputTagPFConversions_<<endl;
  }
  

  // get V0s
  Handle< reco::PFV0Collection > pfV0;
  if( useV0_ ) {
    found = iEvent.getByLabel(inputTagPFV0_, pfV0);
    
    if(!found )
      LogError("PFBlockProducer")<<" cannot get PFV0 : "
				 <<inputTagPFV0_<<endl;
  }


  
  // get ECAL, HCAL, HO and PS clusters
  
  
  Handle< reco::PFClusterCollection > clustersECAL;
  found = iEvent.getByLabel(inputTagPFClustersECAL_, 
			    clustersECAL);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get ECAL clusters: "
			       <<inputTagPFClustersECAL_<<endl;
    
  
  Handle< reco::PFClusterCollection > clustersHCAL;
  found = iEvent.getByLabel(inputTagPFClustersHCAL_, 
			    clustersHCAL);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get HCAL clusters: "
			       <<inputTagPFClustersHCAL_<<endl;
    
  Handle< reco::PFClusterCollection > clustersHO;
  if (useHO_) {
    found = iEvent.getByLabel(inputTagPFClustersHO_, 
			      clustersHO);      
    if(!found )
      LogError("PFBlockProducer")<<" cannot get HO clusters: "
				 <<inputTagPFClustersHO_<<endl;
  }

  Handle< reco::PFClusterCollection > clustersHFEM;
  found = iEvent.getByLabel(inputTagPFClustersHFEM_, 
			    clustersHFEM);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get HFEM clusters: "
			       <<inputTagPFClustersHFEM_<<endl;
    
  Handle< reco::PFClusterCollection > clustersHFHAD;
  found = iEvent.getByLabel(inputTagPFClustersHFHAD_, 
			    clustersHFHAD);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get HFHAD clusters: "
			       <<inputTagPFClustersHFHAD_<<endl;
    

  Handle< reco::PFClusterCollection > clustersPS;
  found = iEvent.getByLabel(inputTagPFClustersPS_, 
			    clustersPS);      
  if(!found )
    LogError("PFBlockProducer")<<" cannot get PS clusters: "
			       <<inputTagPFClustersPS_<<endl;

  // dummy. Not used in the full framework 
  Handle< reco::PFRecTrackCollection > nuclearRecTracks;

  
  Handle< reco::PhotonCollection >  egPhotons;
  found = iEvent.getByLabel(inputTagEGPhotons_,
			    egPhotons);

  if(!found && useEGPhotons_ )
    LogError("PFBlockProducer")<<" cannot get photons" 
			       << inputTagEGPhotons_ << endl;
			       
  Handle< reco::SuperClusterCollection >  sceb;
  Handle< reco::SuperClusterCollection >  scee;
  
  if (useSuperClusters_) {
    found = iEvent.getByLabel(inputTagSCBarrel_,
			      sceb);

    if(!found)
      LogError("PFBlockProducer")<<" cannot get sceb" 
				<< inputTagSCBarrel_ << endl;
	  
				
    
    found = iEvent.getByLabel(inputTagSCEndcap_,
			      scee);

    if(!found)
      LogError("PFBlockProducer")<<" cannot get scee" 
				<< inputTagSCEndcap_ << endl;				       
								
  }
  
  Handle<edm::ValueMap<reco::CaloClusterPtr> > pfclusterassoc;
  if (superClusterMatchByRef_) {
    found = iEvent.getByLabel(inputTagPFClusterAssociationEBEE_,
                              pfclusterassoc);

    if(!found)
      LogError("PFBlockProducer")<<" cannot get PFCluster Association" 
                                << inputTagPFClusterAssociationEBEE_ << endl;
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

  LogDebug("PFBlockProducer")<<"STOP event: "<<iEvent.id().event()
			     <<" in run "<<iEvent.id().run()<<endl;
}
