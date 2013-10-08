#include <vector>
#include "RecoEcal/EgammaClusterProducers/interface/EcalDigiSelector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <TMath.h>
#include <iostream>
#include <set>


using namespace reco;
EcalDigiSelector::EcalDigiSelector(const edm::ParameterSet& ps)
{
  selectedEcalEBDigiCollection_ = ps.getParameter<std::string>("selectedEcalEBDigiCollection");
  selectedEcalEEDigiCollection_ = ps.getParameter<std::string>("selectedEcalEEDigiCollection");

  barrelSuperClusterProducer_ = 
	  consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("barrelSuperClusterProducer"));
  endcapSuperClusterProducer_ = 
	  consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("endcapSuperClusterProducer"));

  
  EcalEBRecHitToken_ = 
	  consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("EcalEBRecHitTag"));
  EcalEERecHitToken_ = 
	  consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("EcalEERecHitTag"));

  EcalEBDigiToken_ = 
	  consumes<EBDigiCollection>(ps.getParameter<edm::InputTag>("EcalEBDigiTag"));
  EcalEEDigiToken_ = 
	  consumes<EEDigiCollection>(ps.getParameter<edm::InputTag>("EcalEEDigiTag"));
  
  cluster_pt_thresh_ = ps.getParameter<double>("cluster_pt_thresh");
  single_cluster_thresh_ = ps.getParameter<double>("single_cluster_thresh");

  nclus_sel_ = ps.getParameter<int>("nclus_sel");
  produces<EBDigiCollection>(selectedEcalEBDigiCollection_);
  produces<EEDigiCollection>(selectedEcalEEDigiCollection_);

}
  
void EcalDigiSelector::produce(edm::Event& evt, const edm::EventSetup& es)
{
  
  //Get BarrelSuperClusters to start.
  edm::Handle<reco::SuperClusterCollection> pBarrelSuperClusters;
  
  evt.getByToken(barrelSuperClusterProducer_, pBarrelSuperClusters);

  const reco::SuperClusterCollection & BarrelSuperClusters = *pBarrelSuperClusters;
  //Got BarrelSuperClusters

  //Get BarrelSuperClusters to start.
  edm::Handle<reco::SuperClusterCollection> pEndcapSuperClusters;
  
  evt.getByToken(endcapSuperClusterProducer_, pEndcapSuperClusters);

  const reco::SuperClusterCollection & EndcapSuperClusters = *pEndcapSuperClusters;
  //Got EndcapSuperClusters

  reco::SuperClusterCollection saveBarrelSuperClusters;
  reco::SuperClusterCollection saveEndcapSuperClusters;
  bool meet_single_thresh = false;
  //Loop over barrel superclusters, and apply threshold
  for (int loop=0;loop<int(BarrelSuperClusters.size());loop++){
    SuperCluster clus1 = BarrelSuperClusters[loop];
    float eta1 = clus1.eta();
    float energy1 = clus1.energy();
    float theta1 = 2*atan(exp(-1.*eta1));
    float cluspt1 = energy1 * sin(theta1);
    if (cluspt1 > cluster_pt_thresh_){
      saveBarrelSuperClusters.push_back(clus1);
      if (cluspt1 > single_cluster_thresh_)
	meet_single_thresh = true;
      
    }
  }

  //Loop over endcap superclusters, and apply threshold
  for (int loop=0;loop<int(EndcapSuperClusters.size());loop++){
    SuperCluster clus1 = EndcapSuperClusters[loop];
    float eta1 = clus1.eta();
    float energy1 = clus1.energy();
    float theta1 = 2*atan(exp(-1.*eta1));
    float cluspt1 = energy1 * sin(theta1);
    if (cluspt1 > cluster_pt_thresh_){
      saveEndcapSuperClusters.push_back(clus1);
      if (cluspt1 > single_cluster_thresh_)
	meet_single_thresh = true;
    }
  }
  
  std::auto_ptr<EBDigiCollection> SEBDigiCol(new EBDigiCollection);
  std::auto_ptr<EEDigiCollection> SEEDigiCol(new EEDigiCollection);
  int TotClus = saveBarrelSuperClusters.size() + saveEndcapSuperClusters.size();

  if (TotClus >= nclus_sel_ || meet_single_thresh){
	
    if (saveBarrelSuperClusters.size() > 0){
      
		
	  edm::ESHandle<CaloTopology> pTopology;
      es.get<CaloTopologyRecord>().get(pTopology);
      const CaloTopology *topology = pTopology.product();	

      //get barrel digi collection
      edm::Handle<EBDigiCollection> pdigis;
      const EBDigiCollection* digis=0;
      evt.getByToken(EcalEBDigiToken_,pdigis);
      digis = pdigis.product(); // get a ptr to the product

	  edm::Handle<EcalRecHitCollection> prechits;
      const EcalRecHitCollection* rechits=0;
      evt.getByToken(EcalEBRecHitToken_,prechits);
      rechits = prechits.product(); // get a ptr to the product


      if ( digis ) {
         std::vector<DetId> saveTheseDetIds;
         //pick out the detids for the 3x3 in each of the selected superclusters
         for (int loop = 0;loop < int(saveBarrelSuperClusters.size());loop++){
           SuperCluster clus1 = saveBarrelSuperClusters[loop];
           CaloClusterPtr bcref = clus1.seed();
           const BasicCluster *bc = bcref.get();
           //Get the maximum detid
           std::pair<DetId, float> EDetty = 
			   EcalClusterTools::getMaximum(*bc,rechits);
           //get the 3x3 array centered on maximum detid.
           std::vector<DetId> detvec = 
			   EcalClusterTools::matrixDetId(topology,EDetty.first, -1, 1, -1, 1);
           //Loop over the 3x3
           for (int ik = 0;ik<int(detvec.size());++ik)
             saveTheseDetIds.push_back(detvec[ik]);
           
         }
         for (int detloop=0; detloop < int(saveTheseDetIds.size());++detloop){
         	EBDetId detL = EBDetId(saveTheseDetIds[detloop]);
           
           for (EBDigiCollection::const_iterator blah = digis->begin();
                blah!=digis->end();blah++){
             
             if (detL == blah->id()){
               EBDataFrame myDigi = (*blah);
               SEBDigiCol->push_back(detL);
           
               EBDataFrame df( SEBDigiCol->back());
               for (int iq =0;iq<myDigi.size();++iq){
                 df.setSample(iq, myDigi.sample(iq).raw());
               }
               //ebcounter++;
             } 
           }
           //if (ebcounter >= int(saveTheseDetIds.size())) break;
         }//loop over dets
    
      }

    }//If barrel superclusters need saving.
    
    
    if (saveEndcapSuperClusters.size() > 0){
 
      edm::ESHandle<CaloTopology> pTopology;
      es.get<CaloTopologyRecord>().get(pTopology);
      const CaloTopology *topology = pTopology.product();	

      //Get endcap rec hit collection
      //get endcap digi collection
      edm::Handle<EEDigiCollection> pdigis;
      const EEDigiCollection* digis=0;
      evt.getByToken(EcalEEDigiToken_,pdigis);
      digis = pdigis.product(); // get a ptr to the product
  
      edm::Handle<EcalRecHitCollection> prechits;
      const EcalRecHitCollection* rechits=0;
      evt.getByToken(EcalEERecHitToken_,prechits);
      rechits = prechits.product(); // get a ptr to the product

      if ( digis ) {
         //std::vector<DetId> saveTheseDetIds;
         std::set<DetId> saveTheseDetIds;
         //pick out the digis for the 3x3 in each of the selected superclusters
         for (int loop = 0;loop < int(saveEndcapSuperClusters.size());loop++){
           SuperCluster clus1 = saveEndcapSuperClusters[loop];
           CaloClusterPtr bcref = clus1.seed();
           const BasicCluster *bc = bcref.get();
           //Get the maximum detid
           std::pair<DetId, float> EDetty = EcalClusterTools::getMaximum(*bc,rechits);
           //get the 3x3 array centered on maximum detid.
           std::vector<DetId> detvec = 
			   EcalClusterTools::matrixDetId(topology,EDetty.first, -1, 1, -1, 1);
           //Loop over the 3x3
           for (int ik = 0;ik<int(detvec.size());++ik)
             saveTheseDetIds.insert(detvec[ik]);
         }
         int eecounter=0;
         for (EEDigiCollection::const_iterator blah = digis->begin();
              blah!=digis->end();blah++){
           std::set<DetId>::const_iterator finder = saveTheseDetIds.find(blah->id());
           if (finder!=saveTheseDetIds.end()){
             EEDetId detL = EEDetId(*finder);

             if (detL == blah->id()){
               EEDataFrame myDigi = (*blah);
               SEEDigiCol->push_back(detL);
               EEDataFrame df( SEEDigiCol->back());
               for (int iq =0;iq<myDigi.size();++iq){
                 df.setSample(iq, myDigi.sample(iq).raw());	      
               }
               eecounter++;
               
             }
           }
           if (eecounter >= int(saveTheseDetIds.size())) break;
         }//loop over digis
      }
    }//If endcap superclusters need saving.
    
  }//If we're actually saving stuff
  
  //Okay, either my collections have been filled with the requisite Digis, or they haven't.
  

  //Empty collection, or full, still put in event.
  SEBDigiCol->sort();
  SEEDigiCol->sort();
  evt.put(SEBDigiCol, selectedEcalEBDigiCollection_);
  evt.put(SEEDigiCol, selectedEcalEEDigiCollection_);
  
}
