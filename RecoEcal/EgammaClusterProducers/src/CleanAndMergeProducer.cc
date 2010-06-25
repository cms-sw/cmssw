// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/CleanAndMergeProducer.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"


/*
  CleanAndMergeProducer:
  ^^^^^^^^^^^^^^^^^^^^^^

  Takes  as  input the cleaned  and the  uncleaned collection of SC
  and produces an uncleaned collection of SC without the duplicates
  and  a collection of  references  to the cleaned collection of SC

  25 June 2010
  Nikolaos Rompotis and Chris Seez  - Imperial College London
  many thanks to Shahram Rahatlou and Federico Ferri


*/


CleanAndMergeProducer::CleanAndMergeProducer(const edm::ParameterSet& ps)
{
  //
  // The debug level
  std::string debugString = ps.getParameter<std::string>("debugLevel");
  if      (debugString == "DEBUG")   debugL = HybridClusterAlgo::pDEBUG;
  else if (debugString == "INFO")    debugL = HybridClusterAlgo::pINFO;
  else                               debugL = HybridClusterAlgo::pERROR;

  // get the parameters
  // the cleaned collection:
  cleanBcCollection_ = ps.getParameter<std::string>("cleanBcCollection");
  cleanBcProducer_ = ps.getParameter<std::string>("cleanBcProducer");
  cleanScCollection_ = ps.getParameter<std::string>("cleanScCollection");
  cleanScProducer_ = ps.getParameter<std::string>("cleanScProducer");
  //cleanClShapeAssoc_ = ps.getParameter<std::string>("cleanClShapeAssoc");
  // the uncleaned collection
  uncleanBcCollection_ = ps.getParameter<std::string>("uncleanBcCollection");
  uncleanBcProducer_   = ps.getParameter<std::string>("uncleanBcProducer");
  uncleanScCollection_ = ps.getParameter<std::string>("uncleanScCollection");
  uncleanScProducer_   = ps.getParameter<std::string>("uncleanScProducer");
  //uncleanClShapeAssoc_ = ps.getParameter<std::string>("uncleanClShapeAssoc");
  // the names of the products to be produced:
  bcCollection_ = ps.getParameter<std::string>("bcCollection");
  scCollection_ = ps.getParameter<std::string>("scCollection");
  cShapeCollection_ = ps.getParameter<std::string>("cShapeCollection");
  clShapeAssoc_ = ps.getParameter<std::string>("clShapeAssoc");
  refScCollection_ = ps.getParameter<std::string>("refScCollection");
  // to produce the cluster shape collection:
  std::map<std::string,double> providedParameters;  
  providedParameters.insert(std::make_pair("LogWeighted",ps.getParameter<bool>("posCalc_logweight")));
  providedParameters.insert(std::make_pair("T0_barl",ps.getParameter<double>("posCalc_t0")));
  providedParameters.insert(std::make_pair("W0",ps.getParameter<double>("posCalc_w0")));
  providedParameters.insert(std::make_pair("X0",ps.getParameter<double>("posCalc_x0")));

  hitproducer_ = ps.getParameter<std::string>("ecalhitproducer");
  hitcollection_ =ps.getParameter<std::string>("ecalhitcollection");

  shapeAlgo_ = ClusterShapeAlgo(providedParameters);
  // the products:
  produces< reco::ClusterShapeCollection>(cShapeCollection_);
  produces< reco::BasicClusterCollection >(bcCollection_);
  produces< reco::SuperClusterCollection >(scCollection_);
  produces< reco::BasicClusterShapeAssociationCollection >(clShapeAssoc_);
  produces< reco::SuperClusterRefVector >(refScCollection_);
  
}

CleanAndMergeProducer::~CleanAndMergeProducer() {;}

void CleanAndMergeProducer::produce(edm::Event& evt, 
				      const edm::EventSetup& es)
{
  // get the input collections

  // rechits and geometry: (needed for the cluster shape collection)
  // _____________________________________________________________________________________
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  //  evt.getByType(rhcHandle);
  evt.getByLabel(hitproducer_, hitcollection_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      if (debugL <= HybridClusterAlgo::pINFO)
	std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      return;
    }
  const EcalRecHitCollection *hit_collection = rhcHandle.product();

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *geometry_p;
  std::auto_ptr<const CaloSubdetectorTopology> topology;
  //
  if(hitcollection_ == "EcalRecHitsEB") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    topology.reset(new EcalBarrelTopology(geoHandle));
  } else if(hitcollection_ == "EcalRecHitsEE") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    topology.reset(new EcalEndcapTopology(geoHandle));
  } else if(hitcollection_ == "EcalRecHitsPS") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
    topology.reset(new EcalPreshowerTopology (geoHandle));
  } else throw(std::runtime_error("\n\nHybrid Cluster Producer encountered invalied ecalhitcollection type.\n\n"));
  // ______________________________________________________________________________________
  //
  // cluster collections:
  edm::Handle<reco::BasicClusterCollection> pCleanBC;
  edm::Handle<reco::SuperClusterCollection> pCleanSC;
  //edm::Handle<reco::BasicClusterShapeAssociationCollection> pCleanClShapeAssoc;
  //
  edm::Handle<reco::BasicClusterCollection> pUncleanBC;
  edm::Handle<reco::SuperClusterCollection> pUncleanSC;
  //edm::Handle<reco::BasicClusterShapeAssociationCollection> pUncleanClShapeAssoc;
  //
  // clean collections ________________________________________________________________
  evt.getByLabel(cleanBcProducer_, cleanBcCollection_, pCleanBC);
  if (!(pCleanBC.isValid())) 
    {
      if (debugL <= HybridClusterAlgo::pINFO)
	std::cout << "could not get a handle on the clean Basic Clusters!" << std::endl;
      return;
    }
  const  reco::BasicClusterCollection cleanBS = *(pCleanBC.product());
  //
  evt.getByLabel(cleanScProducer_, cleanScCollection_, pCleanSC);
  if (!(pCleanSC.isValid())) 
    {
      if (debugL <= HybridClusterAlgo::pINFO)
	std::cout << "could not get a handle on the clean Super Clusters!" << std::endl;
      return;
    }
  const  reco::SuperClusterCollection cleanSC = *(pCleanSC.product());

  //
  // unclean collections _______________________________________________________________
  evt.getByLabel(uncleanBcProducer_, uncleanBcCollection_, pUncleanBC);
  if (!(pUncleanBC.isValid())) 
    {
      if (debugL <= HybridClusterAlgo::pINFO)
	std::cout << "could not get a handle on the unclean Basic Clusters!" << std::endl;
      return;
    }
  const  reco::BasicClusterCollection uncleanBC = *(pUncleanBC.product());
  //
  evt.getByLabel(uncleanScProducer_, uncleanScCollection_, pUncleanSC);
  if (!(pUncleanSC.isValid())) 
    {
      if (debugL <= HybridClusterAlgo::pINFO)
	std::cout << "could not get a handle on the unclean Super Clusters!" << std::endl;
      return;
    }
  const  reco::SuperClusterCollection uncleanSC = *(pUncleanSC.product());
  // for the unlcean collection we need the cluster shape association map
  //evt.getByLabel(uncleanClShapeAssoc_, pUncleanClShapeAssoc);
  //if (!(pUncleanClShapeAssoc.isValid())) 
  //  {
  //    if (debugL <= HybridClusterAlgo::pINFO)
  //	std::cout << "could not get a handle on the unclean ClusterShape association!" << std::endl;
  //    return;
  //  }
  //const  reco::BasicClusterShapeAssociationCollection  
  //  uncleanClShapeAssoc = *(pUncleanClShapeAssoc.product());
  
  //
  // collections are all taken now _____________________________________________________
  //
  //
  // the collections to be produced:
  reco::BasicClusterCollection basicClusters;
  reco::SuperClusterCollection superClusters;
  reco::BasicClusterShapeAssociationCollection shapeAssocs;
  reco::SuperClusterRefVector * scRefs = new reco::SuperClusterRefVector;
  std::vector <reco::ClusterShape> ClusVec;
  //
  // run over the uncleaned SC and check how many of them are matched to the cleaned ones
  // if you find a matched one, create a reference to the cleaned collection and store it
  // if you find an unmatched one, then keep all its basic clusters in the basic Clusters 
  // vector
  int uncleanSize = (int) uncleanSC.size();
  int cleanSize = (int) cleanSC.size();
  if (debugL <= HybridClusterAlgo::pDEBUG)
    std::cout << "Size of Clean Collection: " << cleanSize 
	      << ", uncleanSize: " << uncleanSize  << std::endl;
  //
  // keep whether the SC in unique in the uncleaned collection
  std::vector<int> isUncleanOnly;     // 1 if unique in the uncleaned
  std::vector<int> basicClusterOwner; // contains the index of the SC that owns that BS
  std::vector<int> isSeed; // if this basic cluster is a seed it is 1
  for (int isc =0; isc< uncleanSize; ++isc) {
    const reco::SuperCluster unsc = uncleanSC[isc];    
    const std::vector< std::pair<DetId, float> >   uhits = unsc.hitsAndFractions();
    int uhitsSize = (int) uhits.size();
    bool foundTheSame = false;
    for (int jsc=0; jsc < cleanSize; ++jsc) { // loop over the cleaned SC
      const reco::SuperCluster csc = cleanSC[jsc];
      const std::vector< std::pair<DetId, float> >   chits = csc.hitsAndFractions();
      int chitsSize = (int) chits.size();
      foundTheSame = true;
      if (unsc.seed()->seed() == csc.seed()->seed() && chitsSize == uhitsSize) { 
	// if the clusters are exactly the same then because the clustering
	// algorithm works in a deterministic way, the order of the rechits
	// will be the same
	for (int i=0; i< chitsSize; ++i) {
	  if (uhits[i].first != chits[i].first ) { foundTheSame=false;  break;}
	}
	if (foundTheSame) { // ok you have found it!
	  // make the reference
	  scRefs->push_back( edm::Ref<reco::SuperClusterCollection>(pCleanSC, jsc) );
	  isUncleanOnly.push_back(0);
	  break;
	}
      }
    }
    if (not foundTheSame) {
      // mark it as unique in the uncleaned
      isUncleanOnly.push_back(1);
      // keep all its basic clusters
      reco::CaloCluster_iterator bciter = unsc.clustersBegin();
      for (; bciter != unsc.clustersEnd(); ++bciter) {
	// the basic clusters
	reco::CaloClusterPtr myclusterptr = *bciter;
	reco::CaloCluster mycluster = *myclusterptr;
	basicClusters.push_back(mycluster);
	basicClusterOwner.push_back(isc);
	/*	
	// the cluster shapes
	//
	// for this particular cluster find its index in the unclean BC collection
	int clIndex = -1, counter=-1;
	reco::BasicClusterCollection::const_iterator unbcIter = uncleanBC.begin();
	for (; unbcIter != uncleanBC.end(); ++unbcIter) {
	  ++counter;
	  if (bciter->seed() == unbcIter->seed()) {
	    clIndex = counter;
	    break;
	  }
	}
	if (clIndex == -1) {
	  if (debugL <= HybridClusterAlgo::pINFO)
	    std::cout << "could not match basic cluster from uncleaned collection!" << std::endl;
	  return;
	}
	edm::Ref< reco::BasicClusterCollection > clusterRef(pUncleanBC, clIndex);
	reco::BasicClusterShapeAssociationCollection::const_iterator 
	  map = (*pUncleanClShapeAssoc).find( clusterRef );
	ClusVec.insert( ... , ...);
	*/
      }
    }
  }
  int bcSize = (int) basicClusters.size();
  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Found cleaned SC: " << cleanSize <<  " uncleaned SC: " << uncleanSize << " from which "
	      << scRefs->size() << " will become refs to the cleaned collection" << std::endl;
  // the cluster shapes
  //


  // the following is how to calculate the cluster shape yourself
  for (int erg=0;erg<int(basicClusters.size());++erg){
    reco::ClusterShape TestShape = 
      shapeAlgo_.Calculate(basicClusters[erg],hit_collection,geometry_p,topology.get());
    ClusVec.push_back(TestShape);
  }

  std::auto_ptr< reco::ClusterShapeCollection> cShapeCollection_p(new reco::ClusterShapeCollection);
  cShapeCollection_p->assign(ClusVec.begin(), ClusVec.end());
  edm::OrphanHandle<reco::ClusterShapeCollection> clusHandle 
    = evt.put(cShapeCollection_p, cShapeCollection_);
  //
  // now you have the collection of basic clusters of the SC to be remain in the
  // in the clean collection, export them to the event
  // you will need to reread them later in order to make correctly the refs to the SC
  std::auto_ptr< reco::BasicClusterCollection> basicClusters_p(new reco::BasicClusterCollection);
  basicClusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =  
    evt.put(basicClusters_p, bcCollection_);
  if (!(bccHandle.isValid())) {
    if (debugL <= HybridClusterAlgo::pINFO)
      std::cout << "could not get a handle on the BasicClusterCollection!" << std::endl;
    return;
  }
  reco::BasicClusterCollection basicClustersProd = *bccHandle;
  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Got the BasicClusters from the event again" << std::endl;
  // now you have to create again your superclusters
  // you run over the uncleaned SC, but now you know which of them are
  // the ones that are needed and which are their basic clusters
  for (int isc=0; isc< uncleanSize; ++isc) {
    if (isUncleanOnly[isc]==1) { // look for sc that are unique in the unclean collection
      // make the basic cluster collection
      reco::CaloClusterPtrVector clusterPtrVector;
      reco::CaloClusterPtr seed; // the seed is the basic cluster with the highest energy
      double energy = -1;
      for (int jbc=0; jbc< bcSize; ++jbc) {
	if (basicClusterOwner[jbc]==isc) {
	  reco::CaloClusterPtr currentClu = reco::CaloClusterPtr(bccHandle, jbc);
	  clusterPtrVector.push_back(currentClu);
	  if (energy< currentClu->energy()) {
	    energy = currentClu->energy(); seed = currentClu;
	  }
	}
      }
      const reco::SuperCluster unsc = uncleanSC[isc]; 
      reco::SuperCluster newSC(unsc.energy(), unsc.position(), seed, clusterPtrVector );
      superClusters.push_back(newSC);
    }

  }
  // export the collection of references to the clean collection
  std::auto_ptr< reco::SuperClusterRefVector >  scRefs_p( scRefs );
  //  scRefs_p->assign(scRefs.begin(), scRefs.end());
  evt.put(scRefs_p, refScCollection_);

  // the collection of basic clusters is already in the event
  // the collection of cluster shape is already in the event
  // the collection of uncleaned SC
  std::auto_ptr< reco::SuperClusterCollection > superClusters_p(new reco::SuperClusterCollection);
  superClusters_p->assign(superClusters.begin(), superClusters.end());
  evt.put(superClusters_p, scCollection_);
  // the cluster shape association
  
    // BasicClusterShapeAssociationMap
  std::auto_ptr<reco::BasicClusterShapeAssociationCollection> 
    clShapeAssoc_p(new reco::BasicClusterShapeAssociationCollection);
  for (unsigned int i = 0; i < basicClustersProd.size(); i++){
    clShapeAssoc_p->insert(edm::Ref<reco::BasicClusterCollection>(bccHandle,i),
			  edm::Ref<reco::ClusterShapeCollection>(clusHandle,i));
  }  
  
  evt.put(clShapeAssoc_p, clShapeAssoc_);

  if (debugL == HybridClusterAlgo::pDEBUG)
    std::cout << "Hybrid Clusters (Basic/Super) added to the Event! :-)" << std::endl;


}

