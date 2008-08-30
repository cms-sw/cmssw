#include "Calibration/HcalAlCaRecoProducers/interface/AlCaDiJetsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;
using namespace std;
using namespace reco;

namespace cms
{

AlCaDiJetsProducer::AlCaDiJetsProducer(const edm::ParameterSet& iConfig)
{
   m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","generalTracks"); 
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   mInputCalo_ = iConfig.getParameter<std::vector<edm::InputTag> >("srcCalo");
   hbheInput_ = iConfig.getParameter<edm::InputTag>("hbheInput");
   hoInput_ = iConfig.getParameter<edm::InputTag>("hoInput");
   hfInput_ = iConfig.getParameter<edm::InputTag>("hfInput"); 
   allowMissingInputs_ = true;
//register your products
   produces<reco::TrackCollection>("DiJetsTracksCollection");
   produces<CaloJetCollection>("DiJetsBackToBackCollection");
   produces<EcalRecHitCollection>("DiJetsEcalRecHitCollection");
   produces<HBHERecHitCollection>("DiJetsHBHERecHitCollection");
   produces<HORecHitCollection>("DiJetsHORecHitCollection");
   produces<HFRecHitCollection>("DiJetsHFRecHitCollection");

}
void AlCaDiJetsProducer::beginJob( const edm::EventSetup& iSetup)
{

}

AlCaDiJetsProducer::~AlCaDiJetsProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaDiJetsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
// cout<<" Start produce in AlCaDiJetsProducer "<<endl;
// Jet Collections


   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   geo = pG.product();


   double myvalue = 4.*atan(1.);
   double twomyvalue = 8.*atan(1.);
   CaloJet fJet1,fJet2;
  std::auto_ptr<CaloJetCollection> result (new CaloJetCollection); //Corrected jets
  std::auto_ptr<EcalRecHitCollection> miniDiJetsEcalRecHitCollection(new EcalRecHitCollection);
  std::auto_ptr<HBHERecHitCollection> miniDiJetsHBHERecHitCollection(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> miniDiJetsHORecHitCollection(new HORecHitCollection);
  std::auto_ptr<HFRecHitCollection> miniDiJetsHFRecHitCollection(new HFRecHitCollection);
  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);   


    std::vector<edm::InputTag>::const_iterator ic;
    for (ic=mInputCalo_.begin(); ic!=mInputCalo_.end(); ic++) {
//     cout<<" Read jet collection "<<endl;

//     try {

          edm::Handle<CaloJetCollection> jets;                        //Define Inputs
          iEvent.getByLabel(*ic, jets);                            //Get Inputs

   if(!jets.isValid()){
     LogDebug("") << "AlCaDiJetsProducer: Error! can't get CaloJets product!" << std::endl;
     return ;
   }


          CaloJet fJet1n,fJet2n,fJet3n;
//          cout<<" Number of jets "<<jets->size()<<endl;
	  if( jets->size() < 2 ) 
          {
            iEvent.put( outputTColl, "DiJetsTracksCollection");
            iEvent.put( result, "DiJetsBackToBackCollection");
            iEvent.put( miniDiJetsEcalRecHitCollection,"DiJetsEcalRecHitCollection");
            iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHERecHitCollection");
            iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHORecHitCollection");
            iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFRecHitCollection");

                            return;
          }

//   for (CaloJetCollection::const_iterator track=jets->begin(); track!=jets->end(); track++)
//   {
//       cout<<" Phi "<<(*track).phi()<<" Eta "<<(*track).eta()<<" Et "<<(*track).et()<<endl;
//   }
             fJet1n = (*jets)[0];
             fJet2n = (*jets)[1];

             double phi1=fJet1n.phi();
             double phi2=fJet2n.phi();
             double dphi = fabs(phi1-phi2);
             if (dphi > myvalue) dphi = twomyvalue-dphi;
             double degreedphi = dphi*180./myvalue;
//           cout<<" The angle between two jets "<<degreedphi<<" "<<phi1<<" "<<phi2<<" "<<dphi<<" "<<myvalue<<endl;
           int iejet = 0;
           if( fabs(degreedphi-180) < 30. )
           {
//               cout<<" Jet is found "<<endl;
               iejet = 1;
               fJet1 = (*jets)[0];
               fJet2 = (*jets)[1];
           } // dphi
              else
           {
//            cout<<" Keep empty collection "<<endl;
            iEvent.put( outputTColl, "DiJetsTracksCollection");
            iEvent.put( result, "DiJetsBackToBackCollection");
            iEvent.put( miniDiJetsEcalRecHitCollection,"DiJetsEcalRecHitCollection");
            iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHERecHitCollection");
            iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHORecHitCollection");
            iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFRecHitCollection");
//            cout<<" Final return "<<endl;
                            return;
           }
             result->push_back (fJet1n);
             result->push_back (fJet2n);
             if(jets->size()>2) {fJet3n = (*jets)[2];result->push_back (fJet3n);}
//       }
//        catch (cms::Exception& e) { // can't find it!
//            if (!allowMissingInputs_) {
//	      throw e;
//	    }  
//       }

   } // Jet collection

//   cout<<" Read track collection for accepted events "<<result->size()<<endl;
   if(result->size() == 0) {
     iEvent.put( outputTColl, "DiJetsTracksCollection");
     iEvent.put( result, "DiJetsBackToBackCollection");
     iEvent.put( miniDiJetsEcalRecHitCollection,"DiJetsEcalRecHitCollection");
     iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHERecHitCollection");
     iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHORecHitCollection");
     iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFRecHitCollection");
      return;
   }
//   cout<<" Eta of jets "<<fJet1.eta()<<" "<<fJet2.eta()<<" "<<fJet1.phi()<<" "<<fJet2.phi()<<endl; 
// Track Collection 
//   try{
   edm::Handle<reco::TrackCollection> trackCollection;
   iEvent.getByLabel(m_inputTrackLabel,trackCollection);
   if(!trackCollection.isValid()){
     LogDebug("") << "AlCaDiJetsProducer: Error! can't get trackCollection product!" << std::endl;
     return ;
   }


   const reco::TrackCollection tC = *(trackCollection.product());
//   cout<<" Number of tracks "<<tC.size()<<endl; 
   //Create empty output collections

   for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++)
   {
               
               double deta = track->momentum().eta() - fJet1.eta();  
               double dphi = fabs(track->momentum().phi() - fJet1.phi());
               if (dphi > myvalue) dphi = twomyvalue - dphi;
               double ddir1 = sqrt(deta*deta+dphi*dphi);
               deta = track->momentum().eta() - fJet2.eta();
               dphi = fabs(track->momentum().phi() - fJet2.phi());
               if (dphi > myvalue) dphi = twomyvalue - dphi;
               double ddir2 = sqrt(deta*deta+dphi*dphi);

      if( ddir1 < 1.4  || ddir2 < 1.4)      
      {
         outputTColl->push_back(*track);
      } 
   } // track cycle

//       }
//        catch (cms::Exception& e) { // can't find it!
//            if (!allowMissingInputs_) throw e;
//       }

  // Put Ecal and Hcal RecHits around jet axis

    std::vector<edm::InputTag>::const_iterator i;
    for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) {
//   cout<<" Read ECAL collection "<<endl;
//    try {

      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);
   if(!ec.isValid()){
     LogDebug("") << "AlCaDiJetsProducer: Error! can't get ECAL product!" << std::endl;
     return ;
   }

       for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
                                                recHit != (*ec).end(); ++recHit)
       {
// EcalBarrel = 1, EcalEndcap = 2
           GlobalPoint pos = geo->getPosition(recHit->detid());
           double phihit = pos.phi();
           double etahit = pos.eta();
           double deta = etahit - fJet1.eta();
           double dphi = fabs(phihit - fJet1.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr1 = sqrt(deta*deta+dphi*dphi);
           deta = etahit - fJet2.eta();
           dphi = fabs(phihit - fJet2.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);

           if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsEcalRecHitCollection->push_back(*recHit);

       }  // Ecal rechit cycle

//    } catch (cms::Exception& e) { // can't find it!
//    if (!allowMissingInputs_) throw e;
//    }

    } // Ecal label cycle

//    cout<<" Size of ECAL minicollection "<<miniDiJetsEcalRecHitCollection->size()<<endl;
   
//  try {
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel(hbheInput_,hbhe);
   if(!hbhe.isValid()){
     LogDebug("") << "AlCaDiJetsProducer: Error! can't get hbhe product!" << std::endl;
     return ;
   }

  const HBHERecHitCollection Hithbhe = *(hbhe.product());
//  cout<<" Size of HBHE collection "<<Hithbhe.size()<<endl;
   
  for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
        {
           GlobalPoint pos = geo->getPosition(hbheItr->detid());
           double phihit = pos.phi();
           double etahit = pos.eta();
           double deta = etahit - fJet1.eta();
           double dphi = fabs(phihit - fJet1.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr1 = sqrt(deta*deta+dphi*dphi);
           deta = etahit - fJet2.eta();
           dphi = fabs(phihit - fJet2.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);

         if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsHBHERecHitCollection->push_back(*hbheItr);
        } // HBHE cycle

//    } catch (cms::Exception& e) { // can't find it!
//    if (!allowMissingInputs_) {cout<<"No HBHE collection "<<endl; throw e;}
//    }

//   std::cout<<" Size of mini HCAL collection "<<miniDiJetsHBHERecHitCollection->size()<<std::endl;

//  try{  
   edm::Handle<HORecHitCollection> ho;
   iEvent.getByLabel(hoInput_,ho);
   if(!ho.isValid()){
     LogDebug("") << "AlCaDiJetsProducer: Error! can't get HO product!" << std::endl;
     return ;
   }

   const HORecHitCollection Hitho = *(ho.product());
//   cout<<" Size of HO collection "<<Hitho.size()<<endl;
  for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
        {

          GlobalPoint pos = geo->getPosition(hoItr->detid());
           double phihit = pos.phi();
           double etahit = pos.eta();
           double deta = etahit - fJet1.eta();
           double dphi = fabs(phihit - fJet1.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr1 = sqrt(deta*deta+dphi*dphi);
           deta = etahit - fJet2.eta();
           dphi = fabs(phihit - fJet2.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);
	   
//            cout<<" HO "<<dr1<<" "<<dr2<<" "<<pos.phi()<<" "<<pos.eta()<<endl;
	    

         if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsHORecHitCollection->push_back(*hoItr);
        } // HO cycle

//    } catch (cms::Exception& e) { // can't find it!
//        if (!allowMissingInputs_) {cout<<" No HO collection "<<endl; throw e;}
//    }
//  cout<<" Size of mini HO collection "<<miniDiJetsHORecHitCollection->size()<<endl;
//  try {
  edm::Handle<HFRecHitCollection> hf;
  iEvent.getByLabel(hfInput_,hf);
   if(!hf.isValid()){
     LogDebug("") << "AlCaDiJetsProducer: Error! can't get HF product!" << std::endl;
     return ;
   }


  const HFRecHitCollection Hithf = *(hf.product());
//  cout<<" Size of HF collection "<<Hithf.size()<<endl;
  for(HFRecHitCollection::const_iterator hfItr=Hithf.begin(); hfItr!=Hithf.end(); hfItr++)
      {
          GlobalPoint pos = geo->getPosition(hfItr->detid());
           double phihit = pos.phi();
           double etahit = pos.eta();
           double deta = etahit - fJet1.eta();
           double dphi = fabs(phihit - fJet1.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr1 = sqrt(deta*deta+dphi*dphi);
           deta = etahit - fJet2.eta();
           dphi = fabs(phihit - fJet2.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);

         if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsHFRecHitCollection->push_back(*hfItr);
      } // HF cycle

//    } catch (cms::Exception& e) { // can't find it!
//    if (!allowMissingInputs_) throw e;
//    }
//  cout<<" Size of mini HF collection "<<miniDiJetsHFRecHitCollection->size()<<endl;

  //Put selected information in the event
  iEvent.put( outputTColl, "DiJetsTracksCollection");
//   cout<<" Point 1 "<<endl;
  iEvent.put( result, "DiJetsBackToBackCollection");
//    cout<<" Point 2 "<<endl;

  iEvent.put( miniDiJetsEcalRecHitCollection,"DiJetsEcalRecHitCollection");
//    cout<<" Point 3 "<<endl;

  iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHERecHitCollection");
//    cout<<" Point 3 "<<endl;

  iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHORecHitCollection");
//    cout<<" Point 4 "<<endl;

  iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFRecHitCollection");
//    cout<<" Point 5 "<<endl;

} // Method
} // namespace
