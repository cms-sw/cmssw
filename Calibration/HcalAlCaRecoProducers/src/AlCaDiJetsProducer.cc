#include "Calibration/HcalAlCaRecoProducers/interface/AlCaDiJetsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;
using namespace std;
using namespace reco;

namespace cms
{

AlCaDiJetsProducer::AlCaDiJetsProducer(const edm::ParameterSet& iConfig)
{
   tok_jets_ = consumes<CaloJetCollection>(iConfig.getParameter<edm::InputTag>("jetsInput"));
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));
   tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
   tok_hf_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput")); 
   allowMissingInputs_ = true;

  // fill ecal tokens from input labels
   const unsigned nLabels = ecalLabels_.size();
   for ( unsigned i=0; i != nLabels; i++ ) 
     toks_ecal_.push_back(consumes<EcalRecHitCollection>(ecalLabels_[i]));

//register your products
   produces<CaloJetCollection>("DiJetsBackToBackCollection");
   produces<EcalRecHitCollection>("DiJetsEcalRecHitCollection");
   produces<HBHERecHitCollection>("DiJetsHBHERecHitCollection");
   produces<HORecHitCollection>("DiJetsHORecHitCollection");
   produces<HFRecHitCollection>("DiJetsHFRecHitCollection");

}
void AlCaDiJetsProducer::beginJob()
{
}

AlCaDiJetsProducer::~AlCaDiJetsProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaDiJetsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  double pi = 4.*atan(1.);

   std::auto_ptr<CaloJetCollection> result (new CaloJetCollection); //Corrected jets
   std::auto_ptr<EcalRecHitCollection> miniDiJetsEcalRecHitCollection(new EcalRecHitCollection);
   std::auto_ptr<HBHERecHitCollection> miniDiJetsHBHERecHitCollection(new HBHERecHitCollection);
   std::auto_ptr<HORecHitCollection> miniDiJetsHORecHitCollection(new HORecHitCollection);
   std::auto_ptr<HFRecHitCollection> miniDiJetsHFRecHitCollection(new HFRecHitCollection);

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   const CaloGeometry* geo = pG.product();

  // Jets Collections 

   vector<CaloJet> jetv; 

   CaloJet fJet1, fJet2, fJet3;
   edm::Handle<CaloJetCollection> jets;                
   iEvent.getByToken(tok_jets_, jets);    
   int iflag_select = 0; 
   if(jets->size()>1){
    fJet1 = (*jets)[0];
    fJet2 = (*jets)[1];
    double dphi = fabs(fJet1.phi() - fJet2.phi());  
    if(dphi > pi){dphi = 2*pi - dphi;}
    double degreedphi = dphi*180./pi; 
    if(fabs(degreedphi-180)<30.){iflag_select = 1;}
   }
   if(iflag_select == 1){
     result->push_back(fJet1);
     result->push_back(fJet2);
     jetv.push_back(fJet1); 
     jetv.push_back(fJet2); 
     if(jets->size()>2){
     fJet3 = (*jets)[2]; 
     result->push_back(fJet3);
     jetv.push_back(fJet3);
     }     
   } else {
     iEvent.put( result, "DiJetsBackToBackCollection");
     iEvent.put( miniDiJetsEcalRecHitCollection,"DiJetsEcalRecHitCollection");
     iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHERecHitCollection");
     iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHORecHitCollection");
     iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFRecHitCollection");
     return;
   }  
  
  // Ecal Collections 

   std::vector<edm::EDGetTokenT<EcalRecHitCollection> >::const_iterator i;
   for (i=toks_ecal_.begin(); i!=toks_ecal_.end(); i++) {
   edm::Handle<EcalRecHitCollection> ec;
   iEvent.getByToken(*i,ec);
   for(EcalRecHitCollection::const_iterator ecItr = (*ec).begin();
                                                 ecItr != (*ec).end(); ++ecItr)
        {
// EcalBarrel = 1, EcalEndcap = 2
          GlobalPoint pos = geo->getPosition(ecItr->detid());
          double phihit = pos.phi();
          double etahit = pos.eta();
          int iflag_select = 0;  
          for(unsigned int i=0; i<jetv.size(); i++){
            double deta = fabs(etahit - jetv[i].eta());
            double dphi = fabs(phihit - jetv[i].phi());
            if(dphi > pi) dphi = 2*pi - dphi;
            double dr = sqrt(deta*deta+dphi*dphi);
            if(dr < 1.4){iflag_select = 1;}
          }  
          if(iflag_select==1){miniDiJetsEcalRecHitCollection->push_back(*ecItr);}
       }

   }

  // HB & HE Collections 
   
   edm::Handle<HBHERecHitCollection> hbhe;
   iEvent.getByToken(tok_hbhe_,hbhe);
   for(HBHERecHitCollection::const_iterator hbheItr=hbhe->begin(); 
                                                  hbheItr!=hbhe->end(); hbheItr++)
        {
          GlobalPoint pos = geo->getPosition(hbheItr->detid());
          double phihit = pos.phi();
          double etahit = pos.eta();
          int iflag_select = 0;  
          for(unsigned int i=0; i<jetv.size(); i++){
            double deta = fabs(etahit - jetv[i].eta());
            double dphi = fabs(phihit - jetv[i].phi());
            if(dphi > pi) dphi = 2*pi - dphi;
            double dr = sqrt(deta*deta+dphi*dphi);
            if(dr < 1.4){iflag_select = 1;}
          }  
          if(iflag_select==1){miniDiJetsHBHERecHitCollection->push_back(*hbheItr);}
        }
  

// HO Collections 


   edm::Handle<HORecHitCollection> ho;
   iEvent.getByToken(tok_ho_,ho);
   for(HORecHitCollection::const_iterator hoItr=ho->begin(); 
                                                hoItr!=ho->end(); hoItr++)
     {
          GlobalPoint pos = geo->getPosition(hoItr->detid());
          double phihit = pos.phi();
          double etahit = pos.eta();
          int iflag_select = 0;  
          for(unsigned int i=0; i<jetv.size(); i++){
            double deta = fabs(etahit - jetv[i].eta());
            double dphi = fabs(phihit - jetv[i].phi());
            if(dphi > pi) dphi = 2*pi - dphi;
            double dr = sqrt(deta*deta+dphi*dphi);
            if(dr < 1.4){iflag_select = 1;}
          }  
          if(iflag_select==1){miniDiJetsHORecHitCollection->push_back(*hoItr);}
        }
  
  // HF Collection

 
   edm::Handle<HFRecHitCollection> hf;
   iEvent.getByToken(tok_hf_,hf);
   for(HFRecHitCollection::const_iterator hfItr=hf->begin(); 
                                                hfItr!=hf->end(); hfItr++)
       {
          GlobalPoint pos = geo->getPosition(hfItr->detid());
          double phihit = pos.phi();
          double etahit = pos.eta();
          int iflag_select = 0;  
          for(unsigned int i=0; i<jetv.size(); i++){
            double deta = fabs(etahit - jetv[i].eta());
            double dphi = fabs(phihit - jetv[i].phi());
            if(dphi > pi) dphi = 2*pi - dphi;
            double dr = sqrt(deta*deta+dphi*dphi);
            if(dr < 1.4){iflag_select = 1;}
          }  
          if(iflag_select==1){miniDiJetsHFRecHitCollection->push_back(*hfItr);}
       }
 

  //Put selected information in the event

   iEvent.put( result, "DiJetsBackToBackCollection");
   iEvent.put( miniDiJetsEcalRecHitCollection,"DiJetsEcalRecHitCollection");
   iEvent.put( miniDiJetsHBHERecHitCollection, "DiJetsHBHERecHitCollection");
   iEvent.put( miniDiJetsHORecHitCollection, "DiJetsHORecHitCollection");
   iEvent.put( miniDiJetsHFRecHitCollection, "DiJetsHFRecHitCollection");
}
}
