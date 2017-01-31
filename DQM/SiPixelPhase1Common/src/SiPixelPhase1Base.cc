// -*- C++ -*-
//
// Class:      SiPixelPhase1Base
//
// Implementations of the class
//
// Original Author: Yi-Mu "Enoch" Chen

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"

// Constructor requires manually looping the trigger flag settings
// Since constructor of GenericTriggerEventFlag requires
// EDConsumerBase class protected member calls
SiPixelPhase1Base::SiPixelPhase1Base( const edm::ParameterSet& iConfig ) :
  DQMEDAnalyzer(),
  HistogramManagerHolder( iConfig )
{
  auto flags = iConfig.getParameter<edm::VParameterSet>( "triggerflags" );

  for( auto& flag : flags ){
    triggerlist.emplace_back( new GenericTriggerEventFlag(flag, consumesCollector(), *this) );
  }
}

// Booking histograms as required by the DQM
void
SiPixelPhase1Base::bookHistograms(
  DQMStore::IBooker&     iBooker,
  edm::Run const&        run,
  edm::EventSetup const& iSetup )
{
  for( HistogramManager& histoman : histo ){
    histoman.book( iBooker, iSetup );
  }

  // Running trigger flag initialization (per run)
  for( auto& trigger : triggerlist ){
    if( trigger->on() ){
      trigger->initRun( run, iSetup );
    }
  }
}

// trigger checking function
bool
SiPixelPhase1Base::checktrigger(
  const edm::Event&      iEvent,
  const edm::EventSetup& iSetup,
  const unsigned         trgidx )
{
  if( !iEvent.isRealData() ) { return rand()%5; } // Failing the 4/5 time for MC (testing)

  // For actual data
  if( !triggerlist.at(trgidx)->on() ) { return true; } // if flag is not on, return true regardless

  return triggerlist.at(trgidx)->accept( iEvent, iSetup );
}
