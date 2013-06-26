#include "RecoEcal/EgammaClusterProducers/interface/ExampleClusterProducer.h"
#include "RecoEcal/EgammaClusterAlgos/interface/ExampleClusterAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

ExampleClusterProducer::ExampleClusterProducer(const edm::ParameterSet& ps) {


  // use onfiguration file to setup input/output collection names
  nMaxPrintout_ = ps.getUntrackedParameter<int>("nMaxPrintout",1);

  hitProducer_   = ps.getParameter<std::string>("hitProducer");
  hitCollection_ = ps.getParameter<std::string>("hitCollection");
  clusterCollection_ = ps.getParameter<std::string>("clusterCollection");


  // configure your algorithm via ParameterSet
  double energyCut = ps.getUntrackedParameter<double>("energyCut",0.);
  int nXtalCut     = ps.getUntrackedParameter<int>("nXtalCut",-1);

  algo_ = new ExampleClusterAlgo(energyCut,nXtalCut);

}

ExampleClusterProducer::~ExampleClusterProducer() {
 delete algo_;
}


void
ExampleClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

  using namespace edm;

  // handle to the product
  Handle< EcalRecHitCollection > pRecHits;

  // fetch the product
  evt.getByLabel( hitProducer_, hitCollection_, pRecHits);
  if (!pRecHits.isValid()) {
    edm::LogError("ExampleClusterProducerError") << "Error! can't get the product " << hitCollection_.c_str() ;
  }

  // pointer to the object in the product
  const EcalRecHitCollection* rechits = pRecHits.product();
  edm::LogInfo("ExampleClusterProducerInfo") << "total #  calibrated rechits: " << rechits->size() ;

  // output collection of basic clusters
  // reco::BasicClusterCollection defined in BasicClusterFwd.h

  // make the clusters by passing rechits to the agorithm
  std::auto_ptr< reco::BasicClusterCollection >  
    clusters(  new reco::BasicClusterCollection(algo_->makeClusters( *rechits )) );

  // put the product in the event
  evt.put( clusters, clusterCollection_ );
}
