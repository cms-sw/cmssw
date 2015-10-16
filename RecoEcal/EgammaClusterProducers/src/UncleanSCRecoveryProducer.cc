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


UncleanSCRecoveryProducer::UncleanSCRecoveryProducer(const edm::ParameterSet& ps):
  cleanBcCollection_(consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("cleanBcCollection"))),
  cleanScCollection_(consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("cleanScCollection"))),
  uncleanBcCollection_(consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("uncleanBcCollection"))),
  uncleanScCollection_(consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("uncleanScCollection"))),
  bcCollection_(ps.getParameter<std::string>("bcCollection")),
  scCollection_(ps.getParameter<std::string>("scCollection"))
{
        // the products:
        produces< reco::BasicClusterCollection >(bcCollection_);
        produces< reco::SuperClusterCollection >(scCollection_);
}


void UncleanSCRecoveryProducer::produce(edm::StreamID, edm::Event& evt, 
                                        const edm::EventSetup& es) const
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
        evt.getByToken(cleanScCollection_, pCleanSC);      
        const  reco::SuperClusterCollection cleanSC = *(pCleanSC.product());

        // unclean collections ______________________________________________________
        evt.getByToken(uncleanBcCollection_, pUncleanBC);
        const  reco::BasicClusterCollection uncleanBC = *(pUncleanBC.product());
        //
        evt.getByToken(uncleanScCollection_, pUncleanSC);
        const  reco::SuperClusterCollection uncleanSC = *(pUncleanSC.product());
        int uncleanSize = pUncleanSC->size();
        int cleanSize =   pCleanSC->size();

        LogTrace("EcalCleaning")  << "Size of Clean Collection: " << cleanSize 
                << ", uncleanSize: " << uncleanSize;

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
                        basicClusters.push_back(**bciter);
                        // index of the unclean SC
                        basicClusterOwner.push_back( std::make_pair(isc,0) ); 
                }
        }
        // loop over the clean: only the ones which are in common with the unclean
        // are taken into account

        std::vector<DetId> scCleanSeedDetId;  // counting the clean
        std::vector<int> isToBeKept;
        for (int isc =0; isc< cleanSize; ++isc) {
                reco::SuperClusterRef cscRef( pCleanSC, isc );    
                scCleanSeedDetId.push_back(cscRef->seed()->seed());
                for (reco::CaloCluster_iterator bciter = cscRef->clustersBegin(); bciter != cscRef->clustersEnd(); ++bciter) {
                        // the basic clusters
                        basicClusters.push_back(**bciter);
                        // index of the clean SC
                        basicClusterOwner.push_back( std::make_pair(isc,1) ); 
                }
                if (cscRef->isInUnclean()) isToBeKept.push_back(1);
                else isToBeKept.push_back(0);
        }
        //
        // now export the basic clusters into the event and get them back
        std::auto_ptr< reco::BasicClusterCollection> basicClusters_p(new reco::BasicClusterCollection);
        basicClusters_p->assign(basicClusters.begin(), basicClusters.end());
        edm::OrphanHandle<reco::BasicClusterCollection> bccHandle =  
                evt.put(basicClusters_p, bcCollection_);
        if (!(bccHandle.isValid())) {

                edm::LogWarning("MissingInput") << "could not handle the new BasicClusters!";
                return;
        }
        reco::BasicClusterCollection basicClustersProd = *bccHandle;

        LogTrace("EcalCleaning") <<"Got the BasicClusters from the event again";
        int bcSize = bccHandle->size();
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
                reco::SuperCluster newSC(unsc.energy(), unsc.position(), seed, clusterPtrVector );
                newSC.setFlags(reco::CaloCluster::uncleanOnly);
                superClusters.push_back(newSC);
        }
        // run over the clean SC: only those who are in common between the
        // clean and unclean collection are kept
        for (int isc=0; isc< cleanSize; ++isc) {
                reco::SuperClusterRef cscRef( pCleanSC, isc); 
                if (not cscRef->isInUnclean()) continue;
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
                reco::SuperCluster newSC(cscRef->energy(), cscRef->position(), 
                                         seed, clusterPtrVector );
                newSC.setFlags(reco::CaloCluster::common);
                superClusters.push_back(newSC);
        }

        std::auto_ptr< reco::SuperClusterCollection> 
                superClusters_p(new reco::SuperClusterCollection);
        superClusters_p->assign(superClusters.begin(), superClusters.end());

        evt.put(superClusters_p, scCollection_);

        LogTrace("EcalCleaning")<<"Clusters (Basic/Super) added to the Event! :-)";

        // ----- debugging ----------
        // print the new collection SC quantities
        // print out the clean collection SC
        LogTrace("EcalCleaning") << "Clean Collection SC ";
        for (int i=0; i < cleanSize; ++i) {
                const reco::SuperCluster csc = cleanSC[i];
                LogTrace("EcalCleaning") << " >>> clean    #" << i << "; Energy: " << csc.energy()
                        << " eta: " << csc.eta() 
                        << " sc seed detid: " << csc.seed()->seed().rawId();
        }
        // the unclean SC
        LogTrace("EcalCleaning") << "Unclean Collection SC ";
        for (int i=0; i < uncleanSize; ++i) {
                const reco::SuperCluster usc = uncleanSC[i];
                LogTrace("EcalCleaning") << " >>> unclean  #" << i << "; Energy: " << usc.energy()
                        << " eta: " << usc.eta() 
                        << " sc seed detid: " << usc.seed()->seed().rawId();
        }
        // the new collection
        LogTrace("EcalCleaning")<<"The new SC clean collection with size "<< superClusters.size();
        for (unsigned int i=0; i <  superClusters.size(); ++i) {
                const reco::SuperCluster nsc = superClusters[i];
                LogTrace("EcalCleaning")<< " >>> newSC    #" << i << "; Energy: " << nsc.energy()
                        << " eta: " << nsc.eta()  << " isClean=" 
                        << nsc.isInClean() << " isUnclean=" << nsc.isInUnclean()
                        << " sc seed detid: " << nsc.seed()->seed().rawId();
        }
}
