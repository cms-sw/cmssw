#include "Calibration/HcalAlCaRecoProducers/interface/AlCaGammaJetProducer.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

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
   m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","generalTracks");
   correctedIslandBarrelSuperClusterCollection_ = iConfig.getParameter<std::string>("correctedIslandBarrelSuperClusterCollection");
   correctedIslandBarrelSuperClusterProducer_   = iConfig.getParameter<std::string>("correctedIslandBarrelSuperClusterProducer");
   correctedIslandEndcapSuperClusterCollection_ = iConfig.getParameter<std::string>("correctedIslandEndcapSuperClusterCollection");
   correctedIslandEndcapSuperClusterProducer_   = iConfig.getParameter<std::string>("correctedIslandEndcapSuperClusterProducer");  
   allowMissingInputs_=iConfig.getUntrackedParameter<bool>("AllowMissingInputs",true);
   
    
   //register your products
   produces<reco::TrackCollection>("GammaJetTracksCollection");
   produces<EcalRecHitCollection>("GammaJetEcalRecHitCollection");
   produces<HBHERecHitCollection>("GammaJetHBHERecHitCollection");
   produces<HORecHitCollection>("GammaJetHORecHitCollection");
   produces<HFRecHitCollection>("GammaJetHFRecHitCollection");
   produces<CaloJetCollection>("GammaJetJetBackToBackCollection");
   produces<reco::SuperClusterCollection>("GammaJetGammaBackToBackCollection");

    
   
}
void AlCaGammaJetProducer::beginJob()
{
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

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   geo = pG.product();

// Produced collections 
  
  std::auto_ptr<reco::SuperClusterCollection> result (new reco::SuperClusterCollection); //Corrected jets
  std::auto_ptr<CaloJetCollection> resultjet (new CaloJetCollection); //Corrected jets
  std::auto_ptr<EcalRecHitCollection> miniEcalRecHitCollection(new EcalRecHitCollection);
  std::auto_ptr<HBHERecHitCollection> miniHBHERecHitCollection(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> miniHORecHitCollection(new HORecHitCollection);
  std::auto_ptr<HFRecHitCollection> miniHFRecHitCollection(new HFRecHitCollection);
  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
   
   
// Get  Corrected supercluster collection
  int nclusb = 0;
  int ncluse = 0;
  reco::SuperClusterCollection::const_iterator maxclusbarrel;
  reco::SuperClusterCollection::const_iterator maxclusendcap;

  double vetmax = -100.;
  Handle<reco::SuperClusterCollection> pCorrectedIslandBarrelSuperClusters;
  iEvent.getByLabel(correctedIslandBarrelSuperClusterProducer_, 
                    correctedIslandBarrelSuperClusterCollection_, 
		    pCorrectedIslandBarrelSuperClusters);  
  if (!pCorrectedIslandBarrelSuperClusters.isValid()) {
    // can't find it!
    if (!allowMissingInputs_) {
      *pCorrectedIslandBarrelSuperClusters;  // will throw the proper exception
    }
  } else {
    const reco::SuperClusterCollection* correctedIslandBarrelSuperClusters = pCorrectedIslandBarrelSuperClusters.product();
    
    // loop over the super clusters and find the highest
    maxclusbarrel = correctedIslandBarrelSuperClusters->begin();
    for(reco::SuperClusterCollection::const_iterator aClus = correctedIslandBarrelSuperClusters->begin();
	aClus != correctedIslandBarrelSuperClusters->end(); aClus++) {
      double vet = aClus->energy()/cosh(aClus->eta());
          cout<<" Barrel supercluster " << vet <<" energy "<<aClus->energy()<<" eta "<<aClus->eta()<<endl;
      if(vet>20.) {
	if(vet > vetmax)
	  {
	    vetmax = vet;
	    maxclusbarrel = aClus;
	    nclusb = 1;
	  }
      }
    }
    
  }

  Handle<reco::SuperClusterCollection> pCorrectedIslandEndcapSuperClusters;
  iEvent.getByLabel(correctedIslandEndcapSuperClusterProducer_, 
                    correctedIslandEndcapSuperClusterCollection_, 
		    pCorrectedIslandEndcapSuperClusters);  
  if (!pCorrectedIslandEndcapSuperClusters.isValid()) {
    // can't find it!
    if (!allowMissingInputs_) {
      *pCorrectedIslandEndcapSuperClusters;  // will throw the proper exception
    }
  } else {
    const reco::SuperClusterCollection* correctedIslandEndcapSuperClusters = pCorrectedIslandEndcapSuperClusters.product();
    
    // loop over the super clusters and find the highest
    maxclusendcap = correctedIslandEndcapSuperClusters->end();
    double vetmaxe = vetmax;
    for(reco::SuperClusterCollection::const_iterator aClus = correctedIslandEndcapSuperClusters->begin();
	aClus != correctedIslandEndcapSuperClusters->end(); aClus++) {
      double vet = aClus->energy()/cosh(aClus->eta());
      //   cout<<" Endcap supercluster " << vet <<" energy "<<aClus->energy()<<" eta "<<aClus->eta()<<endl;
      if(vet>20.) {
	if(vet > vetmaxe)
	  {
	    vetmaxe = vet;
	    maxclusendcap = aClus;
	    ncluse = 1;
	  }
      }
    }
  }
   
  cout<<" Number of gammas "<<nclusb<<" "<<ncluse<<endl;  
  
  if( nclusb == 0 && ncluse == 0 ) {
   
  iEvent.put( outputTColl, "GammaJetTracksCollection");
  iEvent.put( miniEcalRecHitCollection, "GammaJetEcalRecHitCollection");
  iEvent.put( miniHBHERecHitCollection, "GammaJetHBHERecHitCollection");
  iEvent.put( miniHORecHitCollection, "GammaJetHORecHitCollection");
  iEvent.put( miniHFRecHitCollection, "GammaJetHFRecHitCollection");
  iEvent.put( result, "GammaJetGammaBackToBackCollection");
  iEvent.put( resultjet, "GammaJetJetBackToBackCollection");
  
  return;
  }

  double phigamma = -100.;
  double etagamma = -100.;
  if(ncluse == 1)
  {
    result->push_back(*maxclusendcap);
    phigamma = (*maxclusendcap).phi();
    etagamma = (*maxclusendcap).eta();
    
  } else
  {
    result->push_back(*maxclusbarrel);
    phigamma = (*maxclusbarrel).phi();
    etagamma = (*maxclusbarrel).eta();
  }
  
//  cout<<" Size of egamma clusters "<<result->size()<<endl;
   
//  
// Jet Collection
// Find jet in the angle ~ +- 170 degrees
//
    
    double phijet =  -100.;
    double etajet = -100.;
    double phijet0 =  -100.;
    double etajet0 = -100.;
    
    std::vector<edm::InputTag>::const_iterator ic;
    for (ic=mInputCalo.begin(); ic!=mInputCalo.end(); ic++) {

      edm::Handle<reco::CaloJetCollection> jets;
      iEvent.getByLabel(*ic, jets);
      if (!jets.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) {
	  *jets;  // will throw the proper exception
	}
      } else {
       reco::CaloJetCollection::const_iterator jet = jets->begin ();

 //      cout<<" Size of jets "<<jets->size()<<endl;

       if( jets->size() == 0 ) continue;

         int iejet = 0;   
         int numjet = 0;
 
         for (; jet != jets->end (); jet++)
         {
           phijet0 = (*jet).phi();
           etajet0 = (*jet).eta();

// Only 3 jets are kept
           numjet++;
           if(numjet > 3) break;

           cout<<" phi,eta "<< phigamma<<" "<< etagamma<<" "<<phijet0<<" "<<etajet0<<endl;

// Find jet back to gamma
 	   
	   double dphi = fabs(phigamma-phijet0); 
	   if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
           double deta = fabs(etagamma-etajet0);
           double dr = sqrt(dphi*dphi+deta*deta);
           if(dr < 0.5 ) continue;
           resultjet->push_back ((*jet));

	   dphi = dphi*180./(4.*atan(1.));
	   if( fabs(dphi-180) < 30. )
	   {
   //           cout<<" Jet is found "<<endl;
              iejet = 1;
              phijet = (*jet).phi();
              etajet = (*jet).eta();
	   } // dphi
         } //jet collection
         if(iejet == 0) resultjet->clear(); 
      }
    } // Jet collection
     
     if( resultjet->size() == 0 ) {

      iEvent.put( outputTColl, "GammaJetTracksCollection");
      iEvent.put( miniEcalRecHitCollection, "GammaJetEcalRecHitCollection");
      iEvent.put( miniHBHERecHitCollection, "GammaJetHBHERecHitCollection");
      iEvent.put( miniHORecHitCollection, "GammaJetHORecHitCollection");
      iEvent.put( miniHFRecHitCollection, "GammaJetHFRecHitCollection");
      iEvent.put( result, "GammaJetGammaBackToBackCollection");
      iEvent.put( resultjet, "GammaJetJetBackToBackCollection");

     return;
     }
    // cout<<" Accepted event "<<resultjet->size()<<" PHI gamma "<<phigamma<<std::endl;
//
// Add Ecal, Hcal RecHits around Egamma caluster
//

// Load EcalRecHits

    
    std::vector<edm::InputTag>::const_iterator i;
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);
      if (!ec.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) { 
	  cout<<" No ECAL input "<<endl;
	  *ec;  // will throw the proper exception
	}
      } else {
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
	    
	    dphi = fabs(phijet - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    deta = fabs(etajet - etahit); 
	    dr = sqrt(dphi*dphi + deta*deta);
	    if(dr<1.4)  miniEcalRecHitCollection->push_back(*recHit);
	    
	  }
      }
    }

//   cout<<" Ecal is done "<<endl; 

      edm::Handle<HBHERecHitCollection> hbhe;
      iEvent.getByLabel(hbheLabel_,hbhe);
      if (!hbhe.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) {
	  *hbhe;  // will throw the proper exception
	}
      } else {
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
	    dphi = fabs(phijet - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    deta = fabs(etajet - etahit); 
	    dr = sqrt(dphi*dphi + deta*deta);
	    if(dr<1.4)  miniHBHERecHitCollection->push_back(*hbheItr);
	  }
      }

//   cout<<" HBHE is done "<<endl; 
	
      edm::Handle<HORecHitCollection> ho;
      iEvent.getByLabel(hoLabel_,ho);
      if (!ho.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) {
	  *ho;  // will throw the proper exception
	}
      } else {
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
	    dphi = fabs(phijet - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    deta = fabs(etajet - etahit); 
	    dr = sqrt(dphi*dphi + deta*deta);
	    if(dr<1.4)  miniHORecHitCollection->push_back(*hoItr);
	    
	  }
      }

 //  cout<<" HO is done "<<endl; 
   
      edm::Handle<HFRecHitCollection> hf;
      iEvent.getByLabel(hfLabel_,hf);
      if (!hf.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) {
	  *hf;  // will throw the proper exception
	}
      } else {
	const HFRecHitCollection Hithf = *(hf.product());
	//  cout<<" Size of HF collection "<<Hithf.size()<<endl;
	for(HFRecHitCollection::const_iterator hfItr=Hithf.begin(); hfItr!=Hithf.end(); hfItr++)
	  {
	    GlobalPoint pos = geo->getPosition(hfItr->detid());
	    
	    double phihit = pos.phi();
	    double etahit = pos.eta();
	    
	    double dphi = fabs(phigamma - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    double deta = fabs(etagamma - etahit); 
	    double dr = sqrt(dphi*dphi + deta*deta);
	    
	    if(dr<1.)  miniHFRecHitCollection->push_back(*hfItr);
	    dphi = fabs(phijet - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    deta = fabs(etajet - etahit); 
	    dr = sqrt(dphi*dphi + deta*deta);
	    
	    if( dr < 1.4 )  miniHFRecHitCollection->push_back(*hfItr);
	  }
      }

 //  cout<<" Size of mini HF collection "<<miniHFRecHitCollection->size()<<endl;
     

// Track Collection   
      edm::Handle<reco::TrackCollection> trackCollection;
      iEvent.getByLabel(m_inputTrackLabel,trackCollection);
      if (!trackCollection.isValid()) {
	// can't find it!
	if (!allowMissingInputs_) {
	  *trackCollection;  // will throw the proper exception
	}
      } else {
	const reco::TrackCollection tC = *(trackCollection.product());
  
	//Create empty output collections


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
	    
	    if( ddir1 < 1.4  || ddir2 < 1.4)      
	      {
		outputTColl->push_back(*track);
	      } 
	  }
      }
   
  //Put selected information in the event
  
  iEvent.put( outputTColl, "GammaJetTracksCollection");
  iEvent.put( miniEcalRecHitCollection, "GammaJetEcalRecHitCollection");
  iEvent.put( miniHBHERecHitCollection, "GammaJetHBHERecHitCollection");
  iEvent.put( miniHORecHitCollection, "GammaJetHORecHitCollection");
  iEvent.put( miniHFRecHitCollection, "GammaJetHFRecHitCollection");
  iEvent.put( result, "GammaJetGammaBackToBackCollection");
  iEvent.put( resultjet, "GammaJetJetBackToBackCollection");
}
//}
