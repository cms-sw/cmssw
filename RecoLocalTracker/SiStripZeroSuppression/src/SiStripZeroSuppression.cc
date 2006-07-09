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
    SiStripPedestalsService_(conf),
    SiStripNoiseService_(conf){

    edm::LogInfo("SiStripZeroSuppression") << "[SiStripZeroSuppression::SiStripZeroSuppression] Constructing object...";

    produces< edm::DetSetVector<SiStripDigi> > ("fromScopeMode");
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
    
    SiStripZeroSuppressionAlgorithm_.configure(&SiStripPedestalsService_,&SiStripNoiseService_);
  }

  // Functions that gets called by framework every event
  void SiStripZeroSuppression::produce(edm::Event& e, const edm::EventSetup& es)
  {
    edm::LogInfo("SiStripZeroSuppression") << "[SiStripZeroSuppression::produce] Analysing " << e.id();

    std::string rawDigiProducer = conf_.getParameter<std::string>("RawDigiProducer");
    
    // Step A: Get Inputs 
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > ScopeMode;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > VirginRaw;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > ProcessedRaw;    
    
    e.getByLabel(rawDigiProducer,"ScopeMode"     , ScopeMode);
    e.getByLabel(rawDigiProducer,"VirginRaw"     , VirginRaw);
    e.getByLabel(rawDigiProducer,"ProcessedRaw"  , ProcessedRaw);
    
    std::vector< edm::DetSet<SiStripDigi> >    v_smDigis;
    std::vector< edm::DetSet<SiStripDigi> >    v_vrDigis;
    std::vector< edm::DetSet<SiStripDigi> >    v_prDigis;

    v_smDigis.reserve(10000);
    v_vrDigis.reserve(10000);
    v_prDigis.reserve(10000);

    // Step B: Invoke the strip clusterizer algorithm and fill output collection
    SiStripPedestalsService_.setESObjects(es);
    SiStripNoiseService_.setESObjects(es);
    if ( ScopeMode->size() )
      SiStripZeroSuppressionAlgorithm_.run("ScopeMode"   ,*ScopeMode   ,v_smDigis);
    if ( VirginRaw->size() )
      SiStripZeroSuppressionAlgorithm_.run("VirginRaw"   ,*VirginRaw   ,v_vrDigis);
    if ( ProcessedRaw->size() )
      SiStripZeroSuppressionAlgorithm_.run("ProcessedRaw",*ProcessedRaw,v_prDigis);
   
    std::auto_ptr< edm::DetSetVector<SiStripDigi> >    smDigis( new edm::DetSetVector<SiStripDigi>(v_smDigis) );
    std::auto_ptr< edm::DetSetVector<SiStripDigi> >    vrDigis( new edm::DetSetVector<SiStripDigi>(v_vrDigis) );
    std::auto_ptr< edm::DetSetVector<SiStripDigi> >    prDigis( new edm::DetSetVector<SiStripDigi>(v_prDigis) );


    // Step D: write output to file
    e.put(smDigis, "fromScopeMode");
    e.put(vrDigis, "fromVirginRaw");
    e.put(prDigis, "fromProcessedRaw");
  }
}
