#include "Calibration/HcalAlCaRecoProducers/interface/AlCaDiJetsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;

//namespace cms
//{

AlCaDiJetsProducer::AlCaDiJetsProducer(const edm::ParameterSet& iConfig)
{
   m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks"); 
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   mInputCalo = iConfig.getParameter<std::vector<edm::InputTag> >("srcCalo");

//register your products
   produces<reco::TrackCollection>("JetTracksCollection");
   produces<CaloJetCollection>("DijetBackToBackCollection");
   produces<HBHERecHitCollection>("HBHERecHitCollection");
   produces<HORecHitCollection>("HORecHitCollection");
   produces<HFRecHitCollection>("HFRecHitCollection");

}
void AlCaDiJetsProducer::beginJob( const edm::EventSetup& iSetup)
{

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();

}

AlCaDiJetsProducer::~AlCaDiJetsProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaDiJetsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
// Jet Collections
   double myvalue = 4.*atan(1.);
   double twomyvalue = 8.*atan(1.);
   CaloJet fJet1,fJet2;
   auto_ptr<CaloJetCollection> result (new CaloJetCollection); //Corrected jets

    std::vector<edm::InputTag>::const_iterator ic;
    for (ic=mInputCalo.begin(); ic!=mInputCalo.end(); ic++) {
     try {

          edm::Handle<CaloJetCollection> jets;                        //Define Inputs
          iEvent.getByLabel(*ic, jets);                            //Get Inputs
          auto_ptr<CaloJetCollection> result (new CaloJetCollection); //Corrected jets

          CaloJet fJet1n,fJet2n;
          if(jets->size() > 1 )
          {
             fJet1n = (*jets)[0];
             fJet2n = (*jets)[1];

             fJet1 = (*jets)[0];
             fJet2 = (*jets)[1];


             double phi1=fabs(fJet1n.phi());
             double phi2=fabs(fJet2n.phi());
             double dphi = fabs(phi1-phi2);
             if (dphi > myvalue) dphi = twomyvalue-dphi;
             double degreedphi = dphi*180./myvalue;
             if (fabs(degreedphi-180.) > 10. ) return;
             result->push_back (fJet1n);
             result->push_back (fJet2n);
         } else
            {
               return;
            }
       }
        catch (std::exception& e) { // can't find it!
            if (!allowMissingInputs_) throw e;
       }
   } // Jet collection


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
   }

  // Put Ecal and Hcal RecHits around jet axis


  std::auto_ptr<EcalRecHitCollection> miniDiJetsEcalRecHitCollection(new EcalRecHitCollection);
  std::auto_ptr<HBHERecHitCollection> miniDiJetsHBHERecHitCollection(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> miniDiJetsHORecHitCollection(new HORecHitCollection);
  std::auto_ptr<HFRecHitCollection> miniDiJetsHFRecHitCollection(new HFRecHitCollection);

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
           double deta = etahit - fJet1.eta();
           double dphi = fabs(phihit - fJet1.phi());
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr1 = sqrt(deta*deta+dphi*dphi);
           deta = etahit - fJet2.eta();
           dphi = phihit - fJet2.phi();
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);

           if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsEcalRecHitCollection->push_back(*recHit);

       }

    } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
    }
    }

   
  try {
     edm::Handle<HBHERecHitCollection> hbhe;
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

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
           dphi = phihit - fJet2.phi();
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);

         if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsHBHERecHitCollection->push_back(*hbheItr);
        }
    } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
    }
  try{  
   edm::Handle<HORecHitCollection> ho;
  const HORecHitCollection Hitho = *(ho.product());
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
           dphi = phihit - fJet2.phi();
           if (dphi > myvalue) dphi = twomyvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);

         if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsHORecHitCollection->push_back(*hoItr);
        }
    } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
    }
  try {
  edm::Handle<HFRecHitCollection> hf;
  const HFRecHitCollection Hithf = *(hf.product());
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
           dphi = phihit - fJet2.phi();
           if (dphi > myvalue) dphi = myvalue - dphi;
           double dr2 = sqrt(deta*deta+dphi*dphi);

         if( dr1 < 1.4 || dr2 < 1.4 )  miniDiJetsHFRecHitCollection->push_back(*hfItr);
      }
    } catch (std::exception& e) { // can't find it!
    if (!allowMissingInputs_) throw e;
    }

  //Put selected information in the event
  iEvent.put( outputTColl, "JetTracksCollection");
  iEvent.put( result, "DijetBackToBackCollection");
  iEvent.put( miniDiJetsEcalRecHitCollection,"DiJetsEcalRecHitCollection");
  iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHERecHitCollection");
  iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHORecHitCollection");
  iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFRecHitCollection");

}
//}
