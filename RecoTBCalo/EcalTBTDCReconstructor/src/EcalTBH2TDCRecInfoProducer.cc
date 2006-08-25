#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBH2TDCRecInfoProducer.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
    
EcalTBH2TDCRecInfoProducer::EcalTBH2TDCRecInfoProducer(edm::ParameterSet const& ps)
{
  rawInfoCollection_ = ps.getParameter<std::string>("rawInfoCollection");
  rawInfoProducer_   = ps.getParameter<std::string>("rawInfoProducer");
  triggerDataCollection_ = ps.getParameter<std::string>("triggerDataCollection");
  triggerDataProducer_   = ps.getParameter<std::string>("triggerDataProducer");
  recInfoCollection_        = ps.getParameter<std::string>("recInfoCollection");


  double tdcZero = ps.getParameter< double >("tdcZero");
  
  produces<EcalTBTDCRecInfo>(recInfoCollection_);
  
  algo_ = new EcalTBH2TDCRecInfoAlgo(tdcZero);
}

EcalTBH2TDCRecInfoProducer::~EcalTBH2TDCRecInfoProducer() 
{
  if (algo_)
    delete algo_;
}

void EcalTBH2TDCRecInfoProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Get input
   edm::Handle<HcalTBTiming> ecalRawTDC;  
   const HcalTBTiming* ecalTDCRawInfo = 0;

   try {
     //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
     e.getByLabel( rawInfoProducer_, ecalRawTDC);
     ecalTDCRawInfo = ecalRawTDC.product();
   } catch ( std::exception& ex ) {
     //     edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << rawInfoCollection_.c_str() ;
   }

   if (! ecalTDCRawInfo )
     {
       edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << rawInfoCollection_.c_str() ;
       return;
     }


   // Get input
   edm::Handle<HcalTBTriggerData> triggerData;  
   const HcalTBTriggerData* h2TriggerData = 0;
   try {
     //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
     e.getByLabel(triggerDataProducer_, triggerData);
     h2TriggerData = triggerData.product();
   } catch ( std::exception& ex ) {
     //     edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << triggerDataCollection_.c_str() ;
   }
   
   if (! h2TriggerData )
     {
       edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << triggerDataCollection_.c_str();
       return;
     }

   
   if (!h2TriggerData->wasBeamTrigger())
     {
       std::auto_ptr<EcalTBTDCRecInfo> recInfo(new EcalTBTDCRecInfo(-1.));
       e.put(recInfo,recInfoCollection_);
     }
   else
     {
        std::auto_ptr<EcalTBTDCRecInfo> recInfo(new EcalTBTDCRecInfo(algo_->reconstruct(*ecalRawTDC)));
	e.put(recInfo,recInfoCollection_);
     }
   

} 


