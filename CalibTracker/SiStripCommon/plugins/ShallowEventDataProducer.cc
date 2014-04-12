#include "CalibTracker/SiStripCommon/interface/ShallowEventDataProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

ShallowEventDataProducer::ShallowEventDataProducer(const edm::ParameterSet& iConfig) {
  produces <unsigned int> ( "run"      );
  produces <unsigned int> ( "event"    );
  produces <unsigned int> ( "bx"       );
  produces <unsigned int> ( "lumi"       );
  produces <std::vector<bool> > ( "TrigTech" );
  produces <std::vector<bool> > ( "TrigPh" );
}

void ShallowEventDataProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {


  std::auto_ptr<unsigned int >  run   ( new unsigned int(iEvent.id().run()   ) );
  std::auto_ptr<unsigned int >  event ( new unsigned int(iEvent.id().event() ) );
  std::auto_ptr<unsigned int >  bx    ( new unsigned int(iEvent.bunchCrossing() ) );
  std::auto_ptr<unsigned int >  lumi  ( new unsigned int(iEvent.luminosityBlock() ) );


  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByLabel( edm::InputTag("gtDigis"), gtRecord);

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

  iEvent.put( run,      "run"   );
  iEvent.put( event,    "event" );
  iEvent.put( bx,       "bx" );
  iEvent.put( lumi,       "lumi" );
  iEvent.put( TrigTech, "TrigTech" );
  iEvent.put( TrigPh,   "TrigPh" );
}
