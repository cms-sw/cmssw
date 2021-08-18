/*
UnifiedSCCollectionProducer:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Takes  as  input the cleaned  and the  uncleaned collection of SC
and  produces two collections of SC: one with the clean SC, but flagged
such that with the algoID value one can identify the SC that are also
in the unclean collection and a collection with the unclean only SC.
This collection has the algoID enumeration of the SC altered
such that:
flags = 0   (cleanedOnly)     cluster is only in the cleaned collection
flags = 100 (common)          cluster is common in both collections
flags = 200 (uncleanedOnly)   cluster is only in the uncleaned collection

In that way the user can get hold of objects from the
-  cleaned   collection only if they choose flags <  200
-  uncleaned collection only if they choose flags >= 100

18 Aug 2010
Nikolaos Rompotis and Chris Seez  - Imperial College London
many thanks to David Wardrope, Shahram Rahatlou and Federico Ferri
*/

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
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include <iostream>
#include <memory>
#include <vector>

class UnifiedSCCollectionProducer : public edm::stream::EDProducer<> {
public:
  UnifiedSCCollectionProducer(const edm::ParameterSet& ps);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // the clean collection
  edm::EDGetTokenT<reco::BasicClusterCollection> cleanBcCollection_;
  edm::EDGetTokenT<reco::SuperClusterCollection> cleanScCollection_;
  // the uncleaned collection
  edm::EDGetTokenT<reco::BasicClusterCollection> uncleanBcCollection_;
  edm::EDGetTokenT<reco::SuperClusterCollection> uncleanScCollection_;

  // the names of the products to be produced:
  std::string bcCollection_;
  std::string scCollection_;
  std::string bcCollectionUncleanOnly_;
  std::string scCollectionUncleanOnly_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(UnifiedSCCollectionProducer);

UnifiedSCCollectionProducer::UnifiedSCCollectionProducer(const edm::ParameterSet& ps) {
  using reco::BasicClusterCollection;
  using reco::SuperClusterCollection;
  // get the parameters
  // the cleaned collection:
  cleanBcCollection_ = consumes<BasicClusterCollection>(ps.getParameter<edm::InputTag>("cleanBcCollection"));
  cleanScCollection_ = consumes<SuperClusterCollection>(ps.getParameter<edm::InputTag>("cleanScCollection"));

  // the uncleaned collection
  uncleanBcCollection_ = consumes<BasicClusterCollection>(ps.getParameter<edm::InputTag>("uncleanBcCollection"));
  uncleanScCollection_ = consumes<SuperClusterCollection>(ps.getParameter<edm::InputTag>("uncleanScCollection"));

  // the names of the products to be produced:
  //
  // the clean collection: this is as it was before, but labeled
  bcCollection_ = ps.getParameter<std::string>("bcCollection");
  scCollection_ = ps.getParameter<std::string>("scCollection");
  // the unclean only collection: SC unique to the unclean collection
  bcCollectionUncleanOnly_ = ps.getParameter<std::string>("bcCollectionUncleanOnly");
  scCollectionUncleanOnly_ = ps.getParameter<std::string>("scCollectionUncleanOnly");
  // the products:
  produces<reco::BasicClusterCollection>(bcCollection_);
  produces<reco::SuperClusterCollection>(scCollection_);
  produces<reco::BasicClusterCollection>(bcCollectionUncleanOnly_);
  produces<reco::SuperClusterCollection>(scCollectionUncleanOnly_);
}

void UnifiedSCCollectionProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::LogInfo("UnifiedSC") << ">>>>> Entering UnifiedSCCollectionProducer <<<<<";
  // get the input collections
  // __________________________________________________________________________
  //
  // cluster collections:
  edm::Handle<reco::BasicClusterCollection> pCleanBC;
  edm::Handle<reco::SuperClusterCollection> pCleanSC;
  //
  edm::Handle<reco::BasicClusterCollection> pUncleanBC;
  edm::Handle<reco::SuperClusterCollection> pUncleanSC;

  evt.getByToken(cleanScCollection_, pCleanSC);
  evt.getByToken(cleanBcCollection_, pCleanBC);
  evt.getByToken(uncleanBcCollection_, pUncleanBC);
  evt.getByToken(uncleanScCollection_, pUncleanSC);

  // the collections to be produced ___________________________________________
  reco::BasicClusterCollection basicClusters;
  reco::SuperClusterCollection superClusters;
  //
  reco::BasicClusterCollection basicClustersUncleanOnly;
  reco::SuperClusterCollection superClustersUncleanOnly;
  //
  // run over the uncleaned SC and check how many of them are matched to
  // the cleaned ones
  // if you find a matched one, then keep the info that it is matched
  //    along with which clean SC was matched + its basic clusters
  // if you find an unmatched one, keep the info and store its basic clusters
  //
  //
  int uncleanSize = pUncleanSC->size();
  int cleanSize = pCleanSC->size();

  LogTrace("UnifiedSC") << "Size of Clean Collection: " << cleanSize << ", uncleanSize: " << uncleanSize;

  // keep the indices
  std::vector<int> inUncleanOnlyInd;      // counting the unclean
  std::vector<int> inCleanInd;            // counting the unclean
  std::vector<int> inCleanOnlyInd;        // counting the clean
  std::vector<DetId> scUncleanSeedDetId;  // counting the unclean
  std::vector<DetId> scCleanSeedDetId;    // counting the clean
  // ontains the index of the SC that owns that BS
  // first basic cluster index, second: 0 for unclean and 1 for clean
  std::vector<std::pair<int, int> > basicClusterOwner;
  std::vector<std::pair<int, int> > basicClusterOwnerUncleanOnly;
  // if this basic cluster is a seed it is 1
  std::vector<int> uncleanBasicClusterIsSeed;

  // loop over unclean SC _____________________________________________________
  for (int isc = 0; isc < uncleanSize; ++isc) {
    reco::SuperClusterRef unscRef(pUncleanSC, isc);
    const std::vector<std::pair<DetId, float> >& uhits = unscRef->hitsAndFractions();
    int uhitsSize = uhits.size();
    bool foundTheSame = false;
    for (int jsc = 0; jsc < cleanSize; ++jsc) {  // loop over the cleaned SC
      reco::SuperClusterRef cscRef(pCleanSC, jsc);
      const std::vector<std::pair<DetId, float> >& chits = cscRef->hitsAndFractions();
      int chitsSize = chits.size();
      foundTheSame = false;
      if (unscRef->seed()->seed() == cscRef->seed()->seed() && chitsSize == uhitsSize) {
        // if the clusters are exactly the same then because the clustering
        // algorithm works in a deterministic way, the order of the rechits
        // will be the same
        foundTheSame = true;
        for (int i = 0; i < chitsSize; ++i) {
          if (uhits[i].first != chits[i].first) {
            foundTheSame = false;
            break;
          }
        }
      }
      if (foundTheSame) {  // ok you have found it:
        // this supercluster belongs to both collections
        inUncleanOnlyInd.push_back(0);
        inCleanInd.push_back(jsc);  // keeps the index of the clean SC
        scUncleanSeedDetId.push_back(unscRef->seed()->seed());
        //
        // keep its basic clusters:
        for (reco::CaloCluster_iterator bciter = unscRef->clustersBegin(); bciter != unscRef->clustersEnd(); ++bciter) {
          // the basic clusters
          basicClusters.push_back(**bciter);
          // index of the unclean SC
          basicClusterOwner.push_back(std::make_pair(isc, 0));
        }
        break;  // break the loop over unclean sc
      }
    }
    if (not foundTheSame) {  // this SC is only in the unclean collection
      // mark it as unique in the uncleaned
      inUncleanOnlyInd.push_back(1);
      scUncleanSeedDetId.push_back(unscRef->seed()->seed());
      // keep all its basic clusters
      for (reco::CaloCluster_iterator bciter = unscRef->clustersBegin(); bciter != unscRef->clustersEnd(); ++bciter) {
        // the basic clusters
        basicClustersUncleanOnly.push_back(**bciter);
        basicClusterOwnerUncleanOnly.push_back(std::make_pair(isc, 0));
      }
    }
  }  // loop over the unclean SC _______________________________________________
  //
  int inCleanSize = inCleanInd.size();
  //
  // loop over the clean SC, check that are not in common with the unclean
  // ones and then store their SC as before ___________________________________
  for (int jsc = 0; jsc < cleanSize; ++jsc) {
    // check whether this index is already in the common collection
    bool takenAlready = false;
    for (int j = 0; j < inCleanSize; ++j) {
      if (jsc == inCleanInd[j]) {
        takenAlready = true;
        break;
      }
    }
    if (takenAlready) {
      inCleanOnlyInd.push_back(0);
      scCleanSeedDetId.push_back(DetId(0));
      continue;
    }
    inCleanOnlyInd.push_back(1);
    reco::SuperClusterRef cscRef(pCleanSC, jsc);
    scCleanSeedDetId.push_back(cscRef->seed()->seed());
    for (reco::CaloCluster_iterator bciter = cscRef->clustersBegin(); bciter != cscRef->clustersEnd(); ++bciter) {
      // the basic clusters
      basicClusters.push_back(**bciter);
      basicClusterOwner.push_back(std::make_pair(jsc, 1));
    }
  }  // end loop over clean SC _________________________________________________
     //
     //

  // Final check: in the endcap BC may exist that are not associated to SC,
  // we need to recover them as well (e.g. multi5x5 algo)
  // This is should be optimized (SA, 20110621)

  // loop on original clean BC collection and see if the BC is missing from the new one
  for (reco::BasicClusterCollection::const_iterator bc = pCleanBC->begin(); bc != pCleanBC->end(); ++bc) {
    bool foundTheSame = false;
    for (reco::BasicClusterCollection::const_iterator cleanonly_bc = basicClusters.begin();
         cleanonly_bc != basicClusters.end();
         ++cleanonly_bc) {
      const std::vector<std::pair<DetId, float> >& chits = bc->hitsAndFractions();
      int chitsSize = chits.size();

      const std::vector<std::pair<DetId, float> >& uhits = cleanonly_bc->hitsAndFractions();
      int uhitsSize = uhits.size();

      if (cleanonly_bc->seed() == bc->seed() && chitsSize == uhitsSize) {
        foundTheSame = true;
        for (int i = 0; i < chitsSize; ++i) {
          if (uhits[i].first != chits[i].first) {
            foundTheSame = false;
            break;
          }
        }
      }

    }  // loop on new clean BC collection

    // clean basic cluster is not associated to SC and does not belong to the
    // new collection, add it
    if (!foundTheSame) {
      basicClusters.push_back(*bc);
      LogTrace("UnifiedSC") << "found BC to add that was not associated to any SC";
    }

  }  // loop on original clean BC collection

  // at this point we have the basic cluster collection ready
  // Up to index   basicClusterOwner.size() we have the BC owned by a SC
  // The remaining are BCs not owned by a SC

  int bcSize = (int)basicClusterOwner.size();
  int bcSizeUncleanOnly = (int)basicClustersUncleanOnly.size();

  LogTrace("UnifiedSC") << "Found cleaned SC: " << cleanSize << " uncleaned SC: " << uncleanSize;
  //
  // export the clusters to the event from the clean clusters
  auto basicClusters_p = std::make_unique<reco::BasicClusterCollection>();
  basicClusters_p->assign(basicClusters.begin(), basicClusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle = evt.put(std::move(basicClusters_p), bcCollection_);
  if (!(bccHandle.isValid())) {
    edm::LogWarning("MissingInput") << "could not handle the new BasicClusters!";
    return;
  }
  reco::BasicClusterCollection basicClustersProd = *bccHandle;

  LogTrace("UnifiedSC") << "Got the BasicClusters from the event again";
  //
  // export the clusters to the event: from the unclean only clusters
  auto basicClustersUncleanOnly_p = std::make_unique<reco::BasicClusterCollection>();
  basicClustersUncleanOnly_p->assign(basicClustersUncleanOnly.begin(), basicClustersUncleanOnly.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandleUncleanOnly =
      evt.put(std::move(basicClustersUncleanOnly_p), bcCollectionUncleanOnly_);
  if (!(bccHandleUncleanOnly.isValid())) {
    edm::LogWarning("MissingInput") << "could not handle the new BasicClusters (Unclean Only)!";
    return;
  }
  reco::BasicClusterCollection basicClustersUncleanOnlyProd = *bccHandleUncleanOnly;
  LogTrace("UnifiedSC") << "Got the BasicClusters from the event again  (Unclean Only)";
  //

  // now we can build the SC collection
  //
  // start again from the unclean collection
  // all the unclean SC will become members of the new collection
  // with different algoIDs ___________________________________________________
  for (int isc = 0; isc < uncleanSize; ++isc) {
    reco::CaloClusterPtrVector clusterPtrVector;
    // the seed is the basic cluster with the highest energy
    reco::CaloClusterPtr seed;
    if (inUncleanOnlyInd[isc] == 1) {  // unclean SC Unique in Unclean
      for (int jbc = 0; jbc < bcSizeUncleanOnly; ++jbc) {
        std::pair<int, int> theBcOwner = basicClusterOwnerUncleanOnly[jbc];
        if (theBcOwner.first == isc && theBcOwner.second == 0) {
          reco::CaloClusterPtr currentClu = reco::CaloClusterPtr(bccHandleUncleanOnly, jbc);
          clusterPtrVector.push_back(currentClu);
          if (scUncleanSeedDetId[isc] == currentClu->seed()) {
            seed = currentClu;
          }
        }
      }

    } else {  // unclean SC common in clean and unclean
      for (int jbc = 0; jbc < bcSize; ++jbc) {
        std::pair<int, int> theBcOwner = basicClusterOwner[jbc];
        if (theBcOwner.first == isc && theBcOwner.second == 0) {
          reco::CaloClusterPtr currentClu = reco::CaloClusterPtr(bccHandle, jbc);
          clusterPtrVector.push_back(currentClu);
          if (scUncleanSeedDetId[isc] == currentClu->seed()) {
            seed = currentClu;
          }
        }
      }
    }
    //std::cout << "before getting the uncl" << std::endl;
    reco::SuperClusterRef unscRef(pUncleanSC, isc);
    reco::SuperCluster newSC(unscRef->energy(), unscRef->position(), seed, clusterPtrVector);
    // now set the algoID for this SC again
    if (inUncleanOnlyInd[isc] == 1) {
      // set up the quality to unclean only .............
      newSC.setFlags(reco::CaloCluster::uncleanOnly);
      superClustersUncleanOnly.push_back(newSC);
    } else {
      // set up the quality to common  .............
      newSC.setFlags(reco::CaloCluster::common);
      superClusters.push_back(newSC);
    }
    // now you can store your SC

  }  // end loop over unclean SC _______________________________________________
  //  flags numbering scheme
  //  flags =   0 = cleanedOnly     cluster is only in the cleaned collection
  //  flags = 100 = common          cluster is common in both collections
  //  flags = 200 = uncleanedOnly   cluster is only in the uncleaned collection

  // now loop over the clean SC and do the same but now you have to avoid the
  // the duplicated ones ______________________________________________________
  for (int jsc = 0; jsc < cleanSize; ++jsc) {
    //std::cout << "working in cl #" << jsc << std::endl;
    // check that the SC is not in the unclean collection
    if (inCleanOnlyInd[jsc] == 0)
      continue;
    reco::CaloClusterPtrVector clusterPtrVector;
    // the seed is the basic cluster with the highest energy
    reco::CaloClusterPtr seed;
    for (int jbc = 0; jbc < bcSize; ++jbc) {
      std::pair<int, int> theBcOwner = basicClusterOwner[jbc];
      if (theBcOwner.first == jsc && theBcOwner.second == 1) {
        reco::CaloClusterPtr currentClu = reco::CaloClusterPtr(bccHandle, jbc);
        clusterPtrVector.push_back(currentClu);
        if (scCleanSeedDetId[jsc] == currentClu->seed()) {
          seed = currentClu;
        }
      }
    }
    reco::SuperClusterRef cscRef(pCleanSC, jsc);
    reco::SuperCluster newSC(cscRef->energy(), cscRef->position(), seed, clusterPtrVector);
    newSC.setFlags(reco::CaloCluster::cleanOnly);

    // add it to the collection:
    superClusters.push_back(newSC);

  }  // end loop over clean SC _________________________________________________

  LogTrace("UnifiedSC") << "New SC collection was created";

  auto superClusters_p = std::make_unique<reco::SuperClusterCollection>();
  superClusters_p->assign(superClusters.begin(), superClusters.end());

  evt.put(std::move(superClusters_p), scCollection_);

  LogTrace("UnifiedSC") << "Clusters (Basic/Super) added to the Event! :-)";

  auto superClustersUncleanOnly_p = std::make_unique<reco::SuperClusterCollection>();
  superClustersUncleanOnly_p->assign(superClustersUncleanOnly.begin(), superClustersUncleanOnly.end());

  evt.put(std::move(superClustersUncleanOnly_p), scCollectionUncleanOnly_);

  // ----- debugging ----------
  // print the new collection SC quantities

  // print out the clean collection SC
  LogTrace("UnifiedSC") << "Clean Collection SC ";
  for (int i = 0; i < cleanSize; ++i) {
    reco::SuperClusterRef cscRef(pCleanSC, i);
    LogTrace("UnifiedSC") << " >>> clean    #" << i << "; Energy: " << cscRef->energy() << " eta: " << cscRef->eta()
                          << " sc seed detid: " << cscRef->seed()->seed().rawId() << std::endl;
  }
  // the unclean SC
  LogTrace("UnifiedSC") << "Unclean Collection SC ";
  for (int i = 0; i < uncleanSize; ++i) {
    reco::SuperClusterRef uscRef(pUncleanSC, i);
    LogTrace("UnifiedSC") << " >>> unclean  #" << i << "; Energy: " << uscRef->energy() << " eta: " << uscRef->eta()
                          << " sc seed detid: " << uscRef->seed()->seed().rawId();
  }
  // the new collection
  LogTrace("UnifiedSC") << "The new SC clean collection with size " << superClusters.size() << std::endl;

  int new_unclean = 0, new_clean = 0;
  for (int i = 0; i < (int)superClusters.size(); ++i) {
    const reco::SuperCluster& nsc = superClusters[i];
    LogTrace("UnifiedSC") << "SC was got" << std::endl
                          << " ---> energy: " << nsc.energy() << std::endl
                          << " ---> eta: " << nsc.eta() << std::endl
                          << " ---> inClean: " << nsc.isInClean() << std::endl
                          << " ---> id: " << nsc.seed()->seed().rawId() << std::endl
                          << " >>> newSC    #" << i << "; Energy: " << nsc.energy() << " eta: " << nsc.eta()
                          << " isClean=" << nsc.isInClean() << " isUnclean=" << nsc.isInUnclean()
                          << " sc seed detid: " << nsc.seed()->seed().rawId();

    if (nsc.isInUnclean())
      ++new_unclean;
    if (nsc.isInClean())
      ++new_clean;
  }
  LogTrace("UnifiedSC") << "The new SC unclean only collection with size " << superClustersUncleanOnly.size();
  for (int i = 0; i < (int)superClustersUncleanOnly.size(); ++i) {
    const reco::SuperCluster nsc = superClustersUncleanOnly[i];
    LogTrace("UnifiedSC") << " >>> newSC    #" << i << "; Energy: " << nsc.energy() << " eta: " << nsc.eta()
                          << " isClean=" << nsc.isInClean() << " isUnclean=" << nsc.isInUnclean()
                          << " sc seed detid: " << nsc.seed()->seed().rawId();
    if (nsc.isInUnclean())
      ++new_unclean;
    if (nsc.isInClean())
      ++new_clean;
  }
  if ((new_unclean != uncleanSize) || (new_clean != cleanSize)) {
    LogTrace("UnifiedSC") << ">>>>!!!!!! MISMATCH: new unclean/ old unclean= " << new_unclean << " / " << uncleanSize
                          << ", new clean/ old clean" << new_clean << " / " << cleanSize;
  }
}
