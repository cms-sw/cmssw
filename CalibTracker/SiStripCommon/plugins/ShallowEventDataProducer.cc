#include "CalibTracker/SiStripCommon/interface/ShallowEventDataProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

ShallowEventDataProducer::ShallowEventDataProducer(const edm::ParameterSet& iConfig) {
  produces <unsigned int> ( "run"      );
  produces <unsigned int> ( "event"    );
  produces <unsigned int> ( "bx"       );
  produces <unsigned int> ( "lumi"     );
  produces <float>        ( "instLumi" );
  produces <float>        ( "PU"       );
  produces <std::vector<bool> > ( "TrigTech" );
  produces <std::vector<bool> > ( "TrigPh" );

  trig_token_   = consumes<L1GlobalTriggerReadoutRecord>(iConfig.getParameter<edm::InputTag>("trigRecord"));
  scalerToken_  = consumes< LumiScalersCollection >(iConfig.getParameter<edm::InputTag>("lumiScalers"));

}

void ShallowEventDataProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {


  std::auto_ptr<unsigned int >  run   ( new unsigned int(iEvent.id().run()   ) );
  std::auto_ptr<unsigned int >  event ( new unsigned int(iEvent.id().event() ) );
  std::auto_ptr<unsigned int >  bx    ( new unsigned int(iEvent.bunchCrossing() ) );
  std::auto_ptr<unsigned int >  lumi  ( new unsigned int(iEvent.luminosityBlock() ) );


  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByToken(trig_token_, gtRecord);

  std::vector<bool> TrigTech_(64,0);
  std::vector<bool> TrigPh_(128,0);

  // Get dWord after masking disabled bits
  DecisionWord dWord = gtRecord->decisionWord();
  if ( ! dWord.empty() ) { // if board not there this is zero
    // loop over dec. bit to get total rate (no overlap)
    for ( int i = 0; i < 64; ++i ) {
      TrigPh_[i]= dWord[i];
    }
  }
    
  TechnicalTriggerWord tw = gtRecord->technicalTriggerWord();
  if ( ! tw.empty() ) {
    // loop over dec. bit to get total rate (no overlap)
    for ( int i = 0; i < 64; ++i ) {
      TrigTech_[i]=tw[i];
    }
  }

  std::auto_ptr<std::vector<bool> >  TrigTech(new std::vector<bool>(TrigTech_));
  std::auto_ptr<std::vector<bool> > TrigPh(new std::vector<bool>(TrigPh_));
  
  // Luminosity informations
  edm::Handle< LumiScalersCollection > lumiScalers;
  float instLumi_=0; float PU_=0;
  iEvent.getByToken(scalerToken_, lumiScalers);
  if(lumiScalers.isValid()){ 
    if (lumiScalers->begin() != lumiScalers->end()) {
      instLumi_ = lumiScalers->begin()->instantLumi();
      PU_       = lumiScalers->begin()->pileup();
    }
  } else {
    edm::LogInfo("ShallowEventDataProducer") 
      << "LumiScalers collection not found in the event; will write dummy values";
  }
  
  auto instLumi = std::make_unique<float>(instLumi_);
  auto PU       = std::make_unique<float>(PU_);

  iEvent.put( run,      "run"   );
  iEvent.put( event,    "event" );
  iEvent.put( bx,       "bx" );
  iEvent.put( lumi,     "lumi" );
  iEvent.put( TrigTech, "TrigTech" );
  iEvent.put( TrigPh,   "TrigPh" );
  iEvent.put(std::move(instLumi), "instLumi");
  iEvent.put(std::move(PU),       "PU");
}
