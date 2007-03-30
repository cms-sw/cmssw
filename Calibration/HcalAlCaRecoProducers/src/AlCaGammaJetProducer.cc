#include "Calibration/HcalAlCaRecoProducers/interface/AlCaGammaJetProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

using namespace edm;
using namespace std;
using namespace reco;

//namespace cms
//{

AlCaGammaJetProducer::AlCaGammaJetProducer(const edm::ParameterSet& iConfig)
{
   // Take input 
   
   hbheLabel_= iConfig.getParameter<edm::InputTag>("hbheInput");
   hoLabel_=iConfig.getParameter<edm::InputTag>("hoInput");
   hfLabel_=iConfig.getParameter<edm::InputTag>("hfInput");
  
   mInputCalo = iConfig.getParameter<std::vector<edm::InputTag> >("srcCalo");
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   
   m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");
   
   correctedIslandBarrelSuperClusterCollection_ = iConfig.getParameter<std::string>("correctedIslandBarrelSuperClusterCollection");
   correctedIslandBarrelSuperClusterProducer_   = iConfig.getParameter<std::string>("correctedIslandBarrelSuperClusterProducer");
  
   allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",false);
   
    
   //register your products

   produces<reco::TrackCollection>("JetTracksCollection");
   produces<CaloJetCollection>("DijetBackToBackCollection");
}
void AlCaGammaJetProducer::beginJob( const edm::EventSetup& iSetup)
{
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();

}

AlCaGammaJetProducer::~AlCaGammaJetProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaGammaJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
// Get  Corrected supercluster collection
  int nclus = 0;
  reco::SuperClusterCollection::const_iterator maxclusbarrel;
  
   try {
   
  Handle<reco::SuperClusterCollection> pCorrectedIslandBarrelSuperClusters;
  iEvent.getByLabel(correctedIslandBarrelSuperClusterProducer_, 
                    correctedIslandBarrelSuperClusterCollection_, 
		    pCorrectedIslandBarrelSuperClusters);  
  const reco::SuperClusterCollection* correctedIslandBarrelSuperClusters = pCorrectedIslandBarrelSuperClusters.product();
  
  // loop over the super clusters and find the highest
  maxclusbarrel = correctedIslandBarrelSuperClusters->begin();
  double vetmax = -100.;
  for(reco::SuperClusterCollection::const_iterator aClus = correctedIslandBarrelSuperClusters->begin();
                                                           aClus != correctedIslandBarrelSuperClusters->end(); aClus++) {
    double vet = aClus->energy()/cosh(aClus->eta());
    cout<<" Barrel supercluster " << vet <<" energy "<<aClus->energy()<<" eta "<<aClus->eta()<<endl;
    if(vet>20.) {
       if(vet > vetmax)
       {
          vetmax = vet;
	  maxclusbarrel = aClus;
	  nclus = 1;
       }
    }
  }
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
  }
  
  if( nclus == 0 ) return;
  
  auto_ptr<reco::SuperClusterCollection> result (new reco::SuperClusterCollection); //Corrected jets
  result->push_back(*maxclusbarrel);

//  
// Jet Collection
// Find jet in the angle ~ +- 170 degrees
//

    auto_ptr<CaloJetCollection> resultjet (new CaloJetCollection); //Corrected jets
    double phigamma = (*maxclusbarrel).phi();
    double etagamma = (*maxclusbarrel).eta();
    double phijet =  -100.;
    double etajet = -100.;
    
    std::vector<edm::InputTag>::const_iterator ic;
    for (ic=mInputCalo.begin(); ic!=mInputCalo.end(); ic++) {
     try {
       edm::Handle<reco::CaloJetCollection> jets;
       iEvent.getByLabel(*ic, jets);
       reco::CaloJetCollection::const_iterator jet = jets->begin ();
       cout<<" Size of jets "<<jets->size()<<endl;
       if( jets->size() == 0 ) continue;
       
       if(jets->size() > 0 )
       {
         for (; jet != jets->end (); jet++)
         {
           phijet = (*jet).phi();
           etajet = (*jet).eta();
	   if( fabs(etajet) > 1. ) continue; 
	   
	   double dphi = fabs(phigamma-phijet); 
	   if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	   dphi = dphi*180./(4.*atan(1.));
	   if( dphi > 170. )
	   {
//  New collection name	      
	      resultjet->push_back ((*jet));
	   }
	    
         } 
       }   
       } 
	catch (std::exception& e) { // can't find it!
            if (!allowMissingInputs_) throw e;
        }
     } // Jet collection

     if( resultjet->size() == 0 ) return;
     
//
// Add Ecal, Hcal RecHits around Egamma caluster
//

// Load EcalRecHits

    std::auto_ptr<EcalRecHitCollection> miniEcalRecHitCollection(new EcalRecHitCollection);
    
    std::vector<edm::InputTag>::const_iterator i;
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
    try {

      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);

       for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
                                                recHit != (*ec).end(); ++recHit)
       {
// EcalBarrel = 1, EcalEndcap = 2
           GlobalPoint pos = geo->getPosition(recHit->detid());
           double phihit = pos.phi();
           double etahit = pos.eta();
	   
	   double dphi = fabs(phigamma - phihit); 
	   if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	   double deta = fabs(etagamma - etahit); 
           double dr = sqrt(dphi*dphi + deta*deta);
           if(dr<1.)  miniEcalRecHitCollection->push_back(*recHit);
   
       }

    } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
    }
    }

  std::auto_ptr<HBHERecHitCollection> miniHBHERecHitCollection(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> miniHORecHitCollection(new HORecHitCollection);
  

    try {

      edm::Handle<HBHERecHitCollection> hbhe;
      iEvent.getByLabel(hbheLabel_,hbhe);
  
  
  
  const HBHERecHitCollection Hithbhe = *(hbhe.product());
  for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
        {
           GlobalPoint pos = geo->getPosition(hbheItr->detid());
           double phihit = pos.phi();
           double etahit = pos.eta();
	   
	   double dphi = fabs(phigamma - phihit); 
	   if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	   double deta = fabs(etagamma - etahit); 
           double dr = sqrt(dphi*dphi + deta*deta);
	
	
         if(dr<1.)  miniHBHERecHitCollection->push_back(*hbheItr);
        }
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
  }
	
    try {
      edm::Handle<HORecHitCollection> ho;
      iEvent.getByLabel(hoLabel_,ho);
	
  const HORecHitCollection Hitho = *(ho.product());
  for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
        {
           GlobalPoint pos = geo->getPosition(hoItr->detid());
           double phihit = pos.phi();
           double etahit = pos.eta();
	   
	   double dphi = fabs(phigamma - phihit); 
	   if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	   double deta = fabs(etagamma - etahit); 
           double dr = sqrt(dphi*dphi + deta*deta);
	
         if(dr<1.)  miniHORecHitCollection->push_back(*hoItr);
        }
  } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
  }
     

// Track Collection    
   edm::Handle<reco::TrackCollection> trackCollection;

//   try {
//     iEvent.getByType(trackCollection);
//   } catch ( std::exception& ex ) {
//     LogDebug("") << "AlCaIsoTracksProducer: Error! can't get product!" << std::endl;
//   }
   iEvent.getByLabel(m_inputTrackLabel,trackCollection);
   const reco::TrackCollection tC = *(trackCollection.product());
  
   //Create empty output collections

   std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);

   for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++)
   {
               
               double deta = track->momentum().eta() - etagamma; 
	        
               double dphi = fabs(track->momentum().phi() - phigamma);
	       
               if (dphi > atan(1.)*4.) dphi = 8.*atan(1.) - dphi;
               double ddir1 = sqrt(deta*deta+dphi*dphi);
	       
	       
	       
               deta = track->momentum().eta() - etajet;
               dphi = fabs(track->momentum().phi() - phijet);
               if (dphi > atan(1.)*4.) dphi = 8.*atan(1.) - dphi;
               double ddir2 = sqrt(deta*deta+dphi*dphi);

      if( ddir1 < 0.5  || ddir2 < 0.5)      
      {
         outputTColl->push_back(*track);
      } 
   }
   
  //Put selected information in the event
  
  iEvent.put( outputTColl, "GammaJetTracksCollection");
  iEvent.put( miniEcalRecHitCollection, "miniEcalRecHitCollection");
  iEvent.put( miniHBHERecHitCollection, "miniHBHERecHitCollection");
  iEvent.put( miniHORecHitCollection, "miniHORecHitCollection");
  iEvent.put( result, "GammaBackToBackCollection");
  iEvent.put( resultjet, "JetBackToBackCollection");
}
//}
