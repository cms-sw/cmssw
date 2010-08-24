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
// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/UncleanSCRecoveryProducer.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"


/*
  UncleanSCRecoveryProducer:
  ^^^^^^^^^^^^^^^^^^^^^^^^^^

  takes the collections of flagged clean and unclean only SC 
  (this is the output of UnifiedSCCollectionProducer) and
  recovers the original collection of unclean SC.

  18 Aug 2010
  Nikolaos Rompotis and Chris Seez  - Imperial College London
  many thanks to David Wardrope, Shahram Rahatlou and Federico Ferri
*/


UncleanSCRecoveryProducer::UncleanSCRecoveryProducer(const edm::ParameterSet& ps)
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
        // the uncleaned collection
        uncleanBcCollection_ = ps.getParameter<std::string>("uncleanBcCollection");
        uncleanBcProducer_   = ps.getParameter<std::string>("uncleanBcProducer");
        uncleanScCollection_ = ps.getParameter<std::string>("uncleanScCollection");
        uncleanScProducer_   = ps.getParameter<std::string>("uncleanScProducer");
        // the names of the products to be produced:
        bcCollection_ = ps.getParameter<std::string>("bcCollection");
        scCollection_ = ps.getParameter<std::string>("scCollection");
        // the products:
        produces< reco::BasicClusterCollection >(bcCollection_);
        produces< reco::SuperClusterCollection >(scCollection_);


}

UncleanSCRecoveryProducer::~UncleanSCRecoveryProducer() {;}

void UncleanSCRecoveryProducer::produce(edm::Event& evt, 
                                        const edm::EventSetup& es)
{
        // __________________________________________________________________________
        //
        // cluster collections:
        edm::Handle<reco::BasicClusterCollection> pCleanBC;
        edm::Handle<reco::SuperClusterCollection> pCleanSC;
        //
        edm::Handle<reco::BasicClusterCollection> pUncleanBC;
        edm::Handle<reco::SuperClusterCollection> pUncleanSC;
        // clean collections ________________________________________________________
        evt.getByLabel(cleanBcProducer_, cleanBcCollection_, pCleanBC);
        if (!(pCleanBC.isValid())) 
        {
                if (debugL <= HybridClusterAlgo::pINFO)
                        std::cout << "could not handle clean basic clusters" << std::endl;
                return;
        }
        const  reco::BasicClusterCollection cleanBS = *(pCleanBC.product());
        //
        evt.getByLabel(cleanScProducer_, cleanScCollection_, pCleanSC);
        if (!(pCleanSC.isValid())) 
        {
                if (debugL <= HybridClusterAlgo::pINFO)
                        std::cout << "could not handle clean super clusters" << std::endl;
                return;
        }
        const  reco::SuperClusterCollection cleanSC = *(pCleanSC.product());

        // unclean collections ______________________________________________________
        evt.getByLabel(uncleanBcProducer_, uncleanBcCollection_, pUncleanBC);
        if (!(pUncleanBC.isValid())) 
        {
                if (debugL <= HybridClusterAlgo::pINFO)
                        std::cout << "could not handle unclean Basic Clusters!" << std::endl;
                return;
        }
        const  reco::BasicClusterCollection uncleanBC = *(pUncleanBC.product());
        //
        evt.getByLabel(uncleanScProducer_, uncleanScCollection_, pUncleanSC);
        if (!(pUncleanSC.isValid())) 
        {
                if (debugL <= HybridClusterAlgo::pINFO)
                        std::cout << "could not handle unclean super clusters!" << std::endl;
                return;
        }
        const  reco::SuperClusterCollection uncleanSC = *(pUncleanSC.product());
        int uncleanSize = (int) uncleanSC.size();
        int cleanSize = (int) cleanSC.size();
        if (debugL <= HybridClusterAlgo::pDEBUG)
                std::cout << "Size of Clean Collection: " << cleanSize 
                        << ", uncleanSize: " << uncleanSize  << std::endl;

        // collections are all taken now ____________________________________________
        //
        // the collections to be produced ___________________________________________
        reco::BasicClusterCollection basicClusters;
        reco::SuperClusterCollection superClusters;
        //
        //
        // collect all the basic clusters of the SC that belong to the unclean
        // collection and put them into the basicClusters vector
        // keep the information of which SC they belong to
        //
        // loop over the unclean sc: all SC will join the new collection
        std::vector< std::pair<int, int> > basicClusterOwner; // counting all

        std::vector<DetId> scUncleanSeedDetId;  // counting the unclean
        for (int isc =0; isc< uncleanSize; ++isc) {
                const reco::SuperCluster unsc = uncleanSC[isc];    
                scUncleanSeedDetId.push_back(unsc.seed()->seed());
                reco::CaloCluster_iterator bciter = unsc.clustersBegin();
                for (; bciter != unsc.clustersEnd(); ++bciter) {
                        // the basic clusters
                        reco::CaloClusterPtr myclusterptr = *bciter;
                        reco::CaloCluster mycluster = *myclusterptr;
                        basicClusters.push_back(mycluster);
                        // index of the unclean SC
                        basicClusterOwner.push_back( std::make_pair(isc,0) ); 
                }
        }
        // loop over the clean: only the ones which are in common with the unclean
        // are taken into account

        std::vector<DetId> scCleanSeedDetId;  // counting the clean
        std::vector<int> isToBeKept;
        for (int isc =0; isc< cleanSize; ++isc) {
                const reco::SuperCluster csc = cleanSC[isc];    
                scCleanSeedDetId.push_back(csc.seed()->seed());
                reco::CaloCluster_iterator bciter = csc.clustersBegin();
                for (; bciter != csc.clustersEnd(); ++bciter) {
                        // the basic clusters
                        reco::CaloClusterPtr myclusterptr = *bciter;
                        reco::CaloCluster mycluster = *myclusterptr;
                        basicClusters.push_back(mycluster);
                        // index of the clean SC
                        basicClusterOwner.push_back( std::make_pair(isc,1) ); 
                }
                if (csc.isInUnclean()) isToBeKept.push_back(1);
                else isToBeKept.push_back(0);
        }
        //
        // now export the basic clusters into the event and get them back
        std::auto_ptr< reco::BasicClusterCollection> 
                basicClusters_p(new reco::BasicClusterCollection);
        basicClusters_p->assign(basicClusters.begin(), basicClusters.end());
        edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =  
                evt.put(basicClusters_p, bcCollection_);
        if (!(bccHandle.isValid())) {
                if (debugL <= HybridClusterAlgo::pINFO)
                        std::cout << "could not handle the new BasicClusters!" << std::endl;
                return;
        }
        reco::BasicClusterCollection basicClustersProd = *bccHandle;
        if (debugL == HybridClusterAlgo::pDEBUG)
                std::cout << "Got the BasicClusters from the event again" << std::endl;
        int bcSize = (int) basicClustersProd.size();
        //
        // now we can create the SC collection
        //
        // run over the unclean SC: all to be kept here
        for (int isc=0; isc< uncleanSize; ++isc) {
                reco::CaloClusterPtrVector clusterPtrVector;
                // the seed is the basic cluster with the highest energy
                reco::CaloClusterPtr seed; 
                for (int jbc=0; jbc< bcSize; ++jbc) {
                        std::pair<int,int> theBcOwner = basicClusterOwner[jbc];
                        if (theBcOwner.first == isc && theBcOwner.second == 0) {
                                reco::CaloClusterPtr currentClu=reco::CaloClusterPtr(bccHandle,jbc);
                                clusterPtrVector.push_back(currentClu);
                                if (scUncleanSeedDetId[isc] == currentClu->seed()) {
                                        seed = currentClu;
                                }
                        }
                }
                const reco::SuperCluster unsc = uncleanSC[isc]; 
                reco::SuperCluster newSC(unsc.energy(), unsc.position(), 
                                         seed, clusterPtrVector );
                newSC.setFlags(reco::CaloCluster::uncleanOnly);
                superClusters.push_back(newSC);
        }
        // run over the clean SC: only those who are in common between the
        // clean and unclean collection are kept
        for (int isc=0; isc< cleanSize; ++isc) {
                const reco::SuperCluster csc = cleanSC[isc]; 
                if (not csc.isInUnclean()) continue;
                reco::CaloClusterPtrVector clusterPtrVector;
                // the seed is the basic cluster with the highest energy
                reco::CaloClusterPtr seed; 
                for (int jbc=0; jbc< bcSize; ++jbc) {
                        std::pair<int,int> theBcOwner = basicClusterOwner[jbc];
                        if (theBcOwner.first == isc && theBcOwner.second == 1) {
                                reco::CaloClusterPtr currentClu=reco::CaloClusterPtr(bccHandle,jbc);
                                clusterPtrVector.push_back(currentClu);
                                if (scCleanSeedDetId[isc] == currentClu->seed()) {
                                        seed = currentClu;
                                }
                        }
                }
                reco::SuperCluster newSC(csc.energy(), csc.position(), 
                                         seed, clusterPtrVector );
                newSC.setFlags(reco::CaloCluster::common);
                superClusters.push_back(newSC);
        }

        std::auto_ptr< reco::SuperClusterCollection> 
                superClusters_p(new reco::SuperClusterCollection);
        superClusters_p->assign(superClusters.begin(), superClusters.end());

        evt.put(superClusters_p, scCollection_);
        if (debugL == HybridClusterAlgo::pDEBUG)
                std::cout << "Clusters (Basic/Super) added to the Event! :-)" << std::endl;

        // ----- debugging ----------
        // print the new collection SC quantities
        if (debugL == HybridClusterAlgo::pDEBUG) {
                // print out the clean collection SC
                std::cout << "Clean Collection SC " << std::endl;
                for (int i=0; i < cleanSize; ++i) {
                        const reco::SuperCluster csc = cleanSC[i];
                        std::cout << " >>> clean    #" << i << "; Energy: " << csc.energy()
                                << " eta: " << csc.eta() 
                                << " sc seed detid: " << csc.seed()->seed().rawId()
                                << std::endl;
                }
                // the unclean SC
                std::cout << "Unclean Collection SC " << std::endl;
                for (int i=0; i < uncleanSize; ++i) {
                        const reco::SuperCluster usc = uncleanSC[i];
                        std::cout << " >>> unclean  #" << i << "; Energy: " << usc.energy()
                                << " eta: " << usc.eta() 
                                << " sc seed detid: " << usc.seed()->seed().rawId()
                                << std::endl;
                }
                // the new collection
                std::cout << "The new SC clean collection with size "<< superClusters.size()  << std::endl;
                for (int i=0; i < (int) superClusters.size(); ++i) {
                        const reco::SuperCluster nsc = superClusters[i];
                        std::cout << " >>> newSC    #" << i << "; Energy: " << nsc.energy()
                                << " eta: " << nsc.eta()  << " isClean=" 
                                << nsc.isInClean() << " isUnclean=" << nsc.isInUnclean()
                                << " sc seed detid: " << nsc.seed()->seed().rawId()
                                << std::endl;
                }
        }

}
