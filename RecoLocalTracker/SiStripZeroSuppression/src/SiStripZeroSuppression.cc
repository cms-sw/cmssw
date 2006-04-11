// File: SiStripZeroSuppression.cc
// Description:  see SiStripZeroSuppression.h
// Author:  D. Giordano
//
//--------------------------------------------
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripZeroSuppression.h"


namespace cms
{
  SiStripZeroSuppression::SiStripZeroSuppression(edm::ParameterSet const& conf): 
    conf_(conf),
    SiStripZeroSuppressionAlgorithm_(conf),
    SiStripPedestalsService_(conf){

    edm::LogInfo("SiStripZeroSuppression") << "[SiStripZeroSuppression::SiStripZeroSuppression] Constructing object...";

    //produces< edm::DetSetVector<SiStripDigi> > (fromScopeMode");
    produces< edm::DetSetVector<SiStripDigi> > ("fromVirginRaw");
    produces< edm::DetSetVector<SiStripDigi> > ("fromProcessedRaw");
  }
  
  // Virtual destructor needed.
  SiStripZeroSuppression::~SiStripZeroSuppression() { 
    edm::LogInfo("SiStripZeroSuppression") << "[SiStripZeroSuppression::~SiStripZeroSuppression] Destructing object...";
  }  
  
  //Get at the beginning Calibration data (pedestals)
  void SiStripZeroSuppression::beginJob( const edm::EventSetup& es ) {
    edm::LogInfo("SiStripZeroSuppression") << "[SiStripZeroSuppression::beginJob]";
    
    SiStripPedestalsService_.configure(es);
    SiStripZeroSuppressionAlgorithm_.configure(&SiStripPedestalsService_);
  }

  // Functions that gets called by framework every event
  void SiStripZeroSuppression::produce(edm::Event& e, const edm::EventSetup& es)
  {
    edm::LogInfo("SiStripZeroSuppression") << "[SiStripZeroSuppression::produce] Analysing " << e.id();

    std::string rawDigiProducer = conf_.getParameter<std::string>("RawDigiProducer");
    
    // Step A: Get Inputs 
    //edm::Handle< edm::DetSetVector<SiStripRawDigi> > ScopeMode;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > VirginRaw;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > ProcessedRaw;    
    //e.getByLabel(rawDigiProducer,"ScopeMode"   , ScopeMode);
    e.getByLabel(rawDigiProducer,"VirginRaw"   , VirginRaw);
    e.getByLabel(rawDigiProducer,"ProcessedRaw", ProcessedRaw);

    
    // Step B: create empty output collection
    std::auto_ptr< edm::DetSetVector<SiStripDigi> >    smDigis( new edm::DetSetVector<SiStripDigi> );
    std::auto_ptr< edm::DetSetVector<SiStripDigi> >    vrDigis( new edm::DetSetVector<SiStripDigi> );
    std::auto_ptr< edm::DetSetVector<SiStripDigi> >    prDigis( new edm::DetSetVector<SiStripDigi> );
    
    
    // Step C: Invoke the strip clusterizer algorithm
    SiStripPedestalsService_.setESObjects(es);

//     if ( ScopeMode->size() )
//    SiStripZeroSuppressionAlgorithm_.run("ScopeMode"   ,*ScopeMode   ,*smDigis);
    if ( VirginRaw->size() )
      SiStripZeroSuppressionAlgorithm_.run("VirginRaw"   ,*VirginRaw   ,*vrDigis);
    if ( ProcessedRaw->size() )
      SiStripZeroSuppressionAlgorithm_.run("ProcessedRaw",*ProcessedRaw,*prDigis);
    
    //FIXME for debug purpose -- remove it
    //SiStripZeroSuppressionAlgorithm_.run("ProcessedRaw"   ,*VirginRaw   ,*vrDigis);

    // Step D: write output to file
    //if ( smDigis->size() )
    // e.put(smDigis, "fromScopeMode");
    if ( vrDigis->size() )
      e.put(vrDigis, "fromVirginRaw");
    if ( prDigis->size() )
      e.put(prDigis, "fromProcessedRaw");
  }
}
