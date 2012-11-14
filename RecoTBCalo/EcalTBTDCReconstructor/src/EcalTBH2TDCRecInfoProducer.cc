#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBH2TDCRecInfoProducer.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
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


  std::vector<EcalTBH2TDCRecInfoAlgo::EcalTBH2TDCRanges> tdcRanges;

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters ranges=ps.getParameter<Parameters>("tdcZeros");
  for(Parameters::iterator itRanges = ranges.begin(); itRanges != ranges.end(); ++itRanges) 
    {
      EcalTBH2TDCRecInfoAlgo::EcalTBH2TDCRanges aRange;
      aRange.runRanges.first = itRanges->getParameter<int>("startRun");
      aRange.runRanges.second = itRanges->getParameter<int>("endRun");
      aRange.tdcZero = itRanges->getParameter< double >("tdcZero");
      tdcRanges.push_back(aRange);
    }
  
  produces<EcalTBTDCRecInfo>(recInfoCollection_);
  
  algo_ = new EcalTBH2TDCRecInfoAlgo(tdcRanges);
}

EcalTBH2TDCRecInfoProducer::~EcalTBH2TDCRecInfoProducer() 
{
  if (algo_)
    delete algo_;
}

void EcalTBH2TDCRecInfoProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  int runNumber = e.id().run();
  // Get input
  edm::Handle<HcalTBTiming> ecalRawTDC;  
  const HcalTBTiming* ecalTDCRawInfo = 0;
  
  //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
  e.getByLabel( rawInfoProducer_, ecalRawTDC);
  if (ecalRawTDC.isValid()) {
    ecalTDCRawInfo = ecalRawTDC.product();
  }

  
  if (! ecalTDCRawInfo )
    {
      edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << rawInfoCollection_.c_str() ;
      return;
    }
  
  
  // Get input
  edm::Handle<HcalTBTriggerData> triggerData;  
  const HcalTBTriggerData* h2TriggerData = 0;
  //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
  e.getByLabel(triggerDataProducer_, triggerData);
  if (triggerData.isValid()) {
    h2TriggerData = triggerData.product();
  }
  
  if (! h2TriggerData )
    {
      edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << triggerDataCollection_.c_str();
      return;
    }
  
  
  if (!h2TriggerData->wasBeamTrigger())
    {
      std::auto_ptr<EcalTBTDCRecInfo> recInfo(new EcalTBTDCRecInfo(0.5));
      e.put(recInfo,recInfoCollection_);
    }
   else
     {
       std::auto_ptr<EcalTBTDCRecInfo> recInfo(new EcalTBTDCRecInfo(algo_->reconstruct(runNumber,*ecalRawTDC)));
       e.put(recInfo,recInfoCollection_);
     }
  

} 


