#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include <iostream>
#include <memory>
#include <vector>

class CleanAndMergeProducer : public edm::stream::EDProducer<> {
public:
  CleanAndMergeProducer(const edm::ParameterSet& ps);

  ~CleanAndMergeProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<reco::SuperClusterCollection> cleanScToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> uncleanScToken_;

  // the names of the products to be produced:
  std::string bcCollection_;
  std::string scCollection_;
  std::string refScCollection_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CleanAndMergeProducer);

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

CleanAndMergeProducer::CleanAndMergeProducer(const edm::ParameterSet& ps) {
  // get the parameters
  // the cleaned collection:
  cleanScToken_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("cleanScInputTag"));
  uncleanScToken_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("uncleanScInputTag"));

  // the names of the products to be produced:
  bcCollection_ = ps.getParameter<std::string>("bcCollection");
  scCollection_ = ps.getParameter<std::string>("scCollection");
  refScCollection_ = ps.getParameter<std::string>("refScCollection");

  std::map<std::string, double> providedParameters;
  providedParameters.insert(std::make_pair("LogWeighted", ps.getParameter<bool>("posCalc_logweight")));
  providedParameters.insert(std::make_pair("T0_barl", ps.getParameter<double>("posCalc_t0")));
  providedParameters.insert(std::make_pair("W0", ps.getParameter<double>("posCalc_w0")));
  providedParameters.insert(std::make_pair("X0", ps.getParameter<double>("posCalc_x0")));

  // the products:
  produces<reco::BasicClusterCollection>(bcCollection_);
  produces<reco::SuperClusterCollection>(scCollection_);
  produces<reco::SuperClusterRefVector>(refScCollection_);
}

CleanAndMergeProducer::~CleanAndMergeProducer() { ; }

void CleanAndMergeProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // get the input collections
  // ______________________________________________________________________________________
  //
  // cluster collections:

  edm::Handle<reco::SuperClusterCollection> pCleanSC;
  edm::Handle<reco::SuperClusterCollection> pUncleanSC;

  evt.getByToken(cleanScToken_, pCleanSC);
  evt.getByToken(uncleanScToken_, pUncleanSC);

  //
  // collections are all taken now _____________________________________________________
  //
  //
  // the collections to be produced:
  reco::BasicClusterCollection basicClusters;
  reco::SuperClusterCollection superClusters;
  reco::SuperClusterRefVector* scRefs = new reco::SuperClusterRefVector;

  //
  // run over the uncleaned SC and check how many of them are matched to the cleaned ones
  // if you find a matched one, create a reference to the cleaned collection and store it
  // if you find an unmatched one, then keep all its basic clusters in the basic Clusters
  // vector
  int uncleanSize = pUncleanSC->size();
  int cleanSize = pCleanSC->size();

  LogTrace("EcalCleaning") << "Size of Clean Collection: " << cleanSize << ", uncleanSize: " << uncleanSize
                           << std::endl;
  //
  // keep whether the SC in unique in the uncleaned collection
  std::vector<int> isUncleanOnly;      // 1 if unique in the uncleaned
  std::vector<int> basicClusterOwner;  // contains the index of the SC that owns that BS
  std::vector<int> isSeed;             // if this basic cluster is a seed it is 1
  for (int isc = 0; isc < uncleanSize; ++isc) {
    reco::SuperClusterRef unscRef(pUncleanSC, isc);
    const std::vector<std::pair<DetId, float> >& uhits = unscRef->hitsAndFractions();
    int uhitsSize = uhits.size();
    bool foundTheSame = false;
    for (int jsc = 0; jsc < cleanSize; ++jsc) {  // loop over the cleaned SC
      reco::SuperClusterRef cscRef(pCleanSC, jsc);
      const std::vector<std::pair<DetId, float> >& chits = cscRef->hitsAndFractions();
      int chitsSize = chits.size();
      foundTheSame = true;
      if (unscRef->seed()->seed() == cscRef->seed()->seed() && chitsSize == uhitsSize) {
        // if the clusters are exactly the same then because the clustering
        // algorithm works in a deterministic way, the order of the rechits
        // will be the same
        for (int i = 0; i < chitsSize; ++i) {
          if (uhits[i].first != chits[i].first) {
            foundTheSame = false;
            break;
          }
        }
        if (foundTheSame) {  // ok you have found it!
          // make the reference
          //scRefs->push_back( edm::Ref<reco::SuperClusterCollection>(pCleanSC, jsc) );
          scRefs->push_back(cscRef);
          isUncleanOnly.push_back(0);
          break;
        }
      }
    }
    if (not foundTheSame) {
      // mark it as unique in the uncleaned
      isUncleanOnly.push_back(1);
      // keep all its basic clusters
      for (reco::CaloCluster_iterator bciter = unscRef->clustersBegin(); bciter != unscRef->clustersEnd(); ++bciter) {
        // the basic clusters
        basicClusters.push_back(**bciter);
        basicClusterOwner.push_back(isc);
      }
    }
  }
  int bcSize = basicClusters.size();

  LogDebug("EcalCleaning") << "Found cleaned SC: " << cleanSize << " uncleaned SC: " << uncleanSize << " from which "
                           << scRefs->size() << " will become refs to the cleaned collection";

  // now you have the collection of basic clusters of the SC to be remain in the
  // in the clean collection, export them to the event
  // you will need to reread them later in order to make correctly the refs to the SC
  auto basicClusters_p = std::make_unique<reco::BasicClusterCollection>();
  basicClusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle = evt.put(std::move(basicClusters_p), bcCollection_);
  if (!(bccHandle.isValid())) {
    edm::LogWarning("MissingInput") << "could not get a handle on the BasicClusterCollection!" << std::endl;
    return;
  }

  LogDebug("EcalCleaning") << "Got the BasicClusters from the event again";
  // now you have to create again your superclusters
  // you run over the uncleaned SC, but now you know which of them are
  // the ones that are needed and which are their basic clusters
  for (int isc = 0; isc < uncleanSize; ++isc) {
    if (isUncleanOnly[isc] == 1) {  // look for sc that are unique in the unclean collection
      // make the basic cluster collection
      reco::CaloClusterPtrVector clusterPtrVector;
      reco::CaloClusterPtr seed;  // the seed is the basic cluster with the highest energy
      double energy = -1;
      for (int jbc = 0; jbc < bcSize; ++jbc) {
        if (basicClusterOwner[jbc] == isc) {
          reco::CaloClusterPtr currentClu = reco::CaloClusterPtr(bccHandle, jbc);
          clusterPtrVector.push_back(currentClu);
          if (energy < currentClu->energy()) {
            energy = currentClu->energy();
            seed = currentClu;
          }
        }
      }
      reco::SuperClusterRef unscRef(pUncleanSC, isc);
      reco::SuperCluster newSC(unscRef->energy(), unscRef->position(), seed, clusterPtrVector);
      superClusters.push_back(newSC);
    }
  }
  // export the collection of references to the clean collection
  std::unique_ptr<reco::SuperClusterRefVector> scRefs_p(scRefs);
  evt.put(std::move(scRefs_p), refScCollection_);

  // the collection of basic clusters is already in the event
  // the collection of uncleaned SC
  auto superClusters_p = std::make_unique<reco::SuperClusterCollection>();
  superClusters_p->assign(superClusters.begin(), superClusters.end());
  evt.put(std::move(superClusters_p), scCollection_);
  LogDebug("EcalCleaning") << "Hybrid Clusters (Basic/Super) added to the Event! :-)";
}
