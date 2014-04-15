#include "RecoParticleFlow/PFClusterProducer/plugins/PFHCALSuperClusterProducer.h"

#include <memory>

#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFSuperCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace std;
using namespace edm;

PFHCALSuperClusterProducer::PFHCALSuperClusterProducer(const edm::ParameterSet& iConfig)
{
    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  // parameters for clustering
  

  // access to the collections of clusters:

   inputTagPFClusters_ =  iConfig.getParameter<InputTag>("PFClusters");    
   inputTagPFClustersHO_ =  iConfig.getParameter<InputTag>("PFClustersHO");    
 
   produces<reco::PFClusterCollection>();
   produces<reco::PFSuperClusterCollection>();

}



PFHCALSuperClusterProducer::~PFHCALSuperClusterProducer() {}




void PFHCALSuperClusterProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {
  

  edm::Handle< reco::PFClusterCollection > clustersHandle;
  edm::Handle< reco::PFClusterCollection > clustersHOHandle;
  
  // access the clusters in the event
  bool found = iEvent.getByLabel( inputTagPFClusters_, clustersHandle );  
  bool foundHO = iEvent.getByLabel( inputTagPFClustersHO_, clustersHOHandle );  

  if(!found ) {

    ostringstream err;
    err<<"cannot find clusters: "<<inputTagPFClusters_;
    LogError("PFHCALSuperClusterProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }

  if(!foundHO ) {

    ostringstream err;
    err<<"cannot find HO clusters: "<<inputTagPFClustersHO_;
    LogError("PFHCALSuperClusterProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }

  // do clustering
  hcalSuperClusterAlgo_.doClustering( clustersHandle, clustersHOHandle );
  
  if( verbose_ ) {
    LogInfo("PFHCALSuperClusterProducer")
      <<"  superclusters --------------------------------- "<<endl
      <<hcalSuperClusterAlgo_<<endl;
  }    
  
  // get clusters out of the clustering algorithm 
  // and put them in the event. There is no copy.
  auto_ptr< vector<reco::PFCluster> > outClusters( hcalSuperClusterAlgo_.clusters() ); 
  auto_ptr< vector<reco::PFSuperCluster> > outSuperClusters( hcalSuperClusterAlgo_.superClusters() ); 
  iEvent.put( outClusters );    
  iEvent.put( outSuperClusters );    

}
  
void PFHCALSuperClusterProducer::endJob(){

hcalSuperClusterAlgo_.write();

}


