/*
Original author Grigory Safronov

27/03/09 - compilation from :
HLTrigger/special/src/HLTHcalNoiseFilter.cc
Calibration/HcalAlCaRecoProducers/src/AlCaEcalHcalReadoutsProducer.cc
Calibration/HcalIsolatedTrackReco/src/SubdetFEDSelector.cc

*/


#include "Calibration/HcalAlCaRecoProducers/interface/AlCaHcalNoiseProducer.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/RawDataCollector/interface/RawDataFEDSelector.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


AlCaHcalNoiseProducer::AlCaHcalNoiseProducer(const edm::ParameterSet& iConfig)
{
   JetSource_ = iConfig.getParameter<edm::InputTag>("JetSource");
   MetSource_ = iConfig.getParameter<edm::InputTag>("MetSource");
   TowerSource_ = iConfig.getParameter<edm::InputTag>("TowerSource");
   useMet_ = iConfig.getParameter<bool>("UseMET");
   useJet_ = iConfig.getParameter<bool>("UseJet");
   MetCut_ = iConfig.getParameter<double>("MetCut");
   JetMinE_ = iConfig.getParameter<double>("JetMinE");
   JetHCALminEnergyFraction_ = iConfig.getParameter<double>("JetHCALminEnergyFraction");

   hoLabel_ = iConfig.getParameter<edm::InputTag>("hoInput");
   hfLabel_ = iConfig.getParameter<edm::InputTag>("hfInput");
   hbheLabel_ = iConfig.getParameter<edm::InputTag>("hbheInput");
   ecalLabels_= iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   ecalPSLabel_=iConfig.getParameter<edm::InputTag>("ecalPSInput");
   rawInLabel_=iConfig.getParameter<edm::InputTag>("rawInput");

  
   //register products
   produces<HBHERecHitCollection>("HBHERecHitCollectionFHN");
   produces<HORecHitCollection>("HORecHitCollectionFHN");
   produces<HFRecHitCollection>("HFRecHitCollectionFHN");
   
   produces<EcalRecHitCollection>("EcalRecHitCollectionFHN");
   produces<EcalRecHitCollection>("PSEcalRecHitCollectionFHN");
    
   produces<FEDRawDataCollection>("HcalFEDsFHN");

}


AlCaHcalNoiseProducer::~AlCaHcalNoiseProducer()
{
}


// ------------ method called to produce the data  ------------
void
AlCaHcalNoiseProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   bool acceptEvent=false;

  // filtering basing on HLTrigger/special/src/HLTHcalNoiseFilter.cc:

   bool isAnomalous_BasedOnMET = false;
   bool isAnomalous_BasedOnEnergyFraction=false; 

   if (useMet_)
     {
       edm::Handle <reco::CaloMETCollection> metHandle;
       iEvent.getByLabel(MetSource_, metHandle);
       const reco::CaloMETCollection *metCol = metHandle.product();
       const reco::CaloMET met = metCol->front();
    
       if(met.pt() > MetCut_) isAnomalous_BasedOnMET=true;
     }
       
   if (useJet_)
     {
       edm::Handle<reco::CaloJetCollection> calojetHandle;
       iEvent.getByLabel(JetSource_,calojetHandle);
       
       edm::Handle<CaloTowerCollection> towerHandle;
       iEvent.getByLabel(TowerSource_, towerHandle);

       std::vector<CaloTower> TowerContainer;
       std::vector<reco::CaloJet> JetContainer;
       TowerContainer.clear();
       JetContainer.clear();
       CaloTower seedTower;
       nEvents++;
       for(reco::CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();calojetIter != calojetHandle->end();++calojetIter) {
	 if( ((calojetIter->et())*cosh(calojetIter->eta()) > JetMinE_) && (calojetIter->energyFractionHadronic() > JetHCALminEnergyFraction_) ) {
	   JetContainer.push_back(*calojetIter);
	   double maxTowerE = 0.0;
	   for(CaloTowerCollection::const_iterator kal = towerHandle->begin(); kal != towerHandle->end(); kal++) {
	     double dR = deltaR((*calojetIter).eta(),(*calojetIter).phi(),(*kal).eta(),(*kal).phi());
	     if( (dR < 0.50) && (kal->p() > maxTowerE) ) {
	       maxTowerE = kal->p();
	       seedTower = *kal;
	     }
	   }
	   TowerContainer.push_back(seedTower);
	 }
	 
       }
       if(JetContainer.size() > 0) {
	 nAnomalousEvents++;
	 isAnomalous_BasedOnEnergyFraction = true;
       }
     }
   
   acceptEvent=((useMet_&&isAnomalous_BasedOnMET)||(useJet_&&isAnomalous_BasedOnEnergyFraction));

   ////////////////

    //Create empty output collections
   
  std::auto_ptr<HBHERecHitCollection> miniHBHERecHitCollection(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> miniHORecHitCollection(new HORecHitCollection);
  std::auto_ptr<HFRecHitCollection> miniHFRecHitCollection(new HFRecHitCollection);

  std::auto_ptr<EcalRecHitCollection> outputEColl(new EcalRecHitCollection);
  std::auto_ptr<EcalRecHitCollection> outputESColl(new EcalRecHitCollection);

  std::auto_ptr<FEDRawDataCollection> outputFEDs(new FEDRawDataCollection);


  // if good event get and save all colletions
  if (acceptEvent)
    {
      edm::Handle<HBHERecHitCollection> hbhe;
      edm::Handle<HORecHitCollection> ho;
      edm::Handle<HFRecHitCollection> hf;
      
      iEvent.getByLabel(hbheLabel_,hbhe);
      iEvent.getByLabel(hoLabel_,ho);
      iEvent.getByLabel(hfLabel_,hf);

      edm::Handle<EcalRecHitCollection> pRecHits;
      iEvent.getByLabel(ecalPSLabel_,pRecHits);

      // temporary collection of EB+EE recHits

      std::auto_ptr<EcalRecHitCollection> tmpEcalRecHitCollection(new EcalRecHitCollection);
      
      std::vector<edm::InputTag>::const_iterator i;
      for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) 
	{
	  edm::Handle<EcalRecHitCollection> ec;
	  iEvent.getByLabel(*i,ec);
	  for(EcalRecHitCollection::const_iterator recHit = (*ec).begin(); recHit != (*ec).end(); ++recHit)
	    {
	      tmpEcalRecHitCollection->push_back(*recHit);
	    }
	}    

      //////////

      //////// write HCAL collections:
      const HBHERecHitCollection Hithbhe = *(hbhe.product());
      for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
	{
	  miniHBHERecHitCollection->push_back(*hbheItr);
	}
      const HORecHitCollection Hitho = *(ho.product());
      for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
	{
	  miniHORecHitCollection->push_back(*hoItr);
	}
      
      const HFRecHitCollection Hithf = *(hf.product());
      for(HFRecHitCollection::const_iterator hfItr=Hithf.begin(); hfItr!=Hithf.end(); hfItr++)
	{
	  miniHFRecHitCollection->push_back(*hfItr);
	}
      /////

      ///// write ECAL       
      for (std::vector<EcalRecHit>::const_iterator ehit=tmpEcalRecHitCollection->begin(); ehit!=tmpEcalRecHitCollection->end(); ehit++) 
	{
	  outputEColl->push_back(*ehit);
	}
      /////////

      // write PS
      const EcalRecHitCollection& psrechits = *(pRecHits.product());
      
      for(EcalRecHitCollection::const_iterator i=psrechits.begin(); i!=psrechits.end(); i++) 
	{
	  outputESColl->push_back( *i );
	}
      

      // get HCAL FEDs
      edm::Handle<FEDRawDataCollection> rawIn;
      iEvent.getByLabel(rawInLabel_,rawIn);
      
      std::vector<int> selFEDs;
      for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
	{
	  selFEDs.push_back(i);
	}
      ////////////

      // Copying FEDs :
      const FEDRawDataCollection *rdc=rawIn.product();
      
      //   if ( ( rawData[i].provenance()->processName() != e.processHistory().rbegin()->processName() ) )
      //       continue ; // skip all raw collections not produced by the current process
      
      for ( int j=0; j< FEDNumbering::MAXFEDID; ++j ) 
	{
	  bool rightFED=false;
	  for (uint32_t k=0; k<selFEDs.size(); k++)
	    {
	      if (j==selFEDs[k])
		{
		  rightFED=true;
		}
	    }
	  if (!rightFED) continue;
	  const FEDRawData & fedData = rdc->FEDData(j);
	  size_t size=fedData.size();
	  
	  if ( size > 0 ) 
	    {
	      // this fed has data -- lets copy it
	      FEDRawData & fedDataProd = outputFEDs->FEDData(j);
	      if ( fedDataProd.size() != 0 ) {
		//	   std::cout << " More than one FEDRawDataCollection with data in FED ";
		//	   std::cout << j << " Skipping the 2nd\n";
		continue;
	      }
	      fedDataProd.resize(size);
	      unsigned char *dataProd=fedDataProd.data();
	      const unsigned char *data=fedData.data();
	      for ( unsigned int k=0; k<size; ++k ) {
		dataProd[k]=data[k];
	      }
	    }
	}
      //////////////////////
    }

  //Put selected information in the event
  iEvent.put( miniHBHERecHitCollection, "HBHERecHitCollectionFHN");
  iEvent.put( miniHORecHitCollection, "HORecHitCollectionFHN");
  iEvent.put( miniHFRecHitCollection, "HFRecHitCollectionFHN");
  iEvent.put( outputEColl, "EcalRecHitCollectionFHN"); 
  iEvent.put( outputESColl, "PSEcalRecHitCollectionFHN");
  iEvent.put( outputFEDs, "HcalFEDsFHN"); 
}
