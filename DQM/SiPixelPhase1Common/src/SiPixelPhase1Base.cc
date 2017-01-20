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
  auto histogramlist = iConfig.getParameter<edm::VParameterSet>( "histograms" );

  for( size_t i = 0; i < histogramlist.size(); ++i ){
    auto& settings  = histogramlist.at( i );
    auto& histogram = histo[i];

    auto flags = settings.getParameter<edm::VParameterSet>( "triggerflags" );

    for( auto& flag : flags ){
      histogram.addTriggerFlag( new GenericTriggerEventFlag( flag, consumesCollector(), *this ) );
    }
  }
}

// analyze function class histogram manager per-event initializing
void
SiPixelPhase1Base::analyze( edm::Event const& e, edm::EventSetup const& eSetup )
{
  for( auto& histoman : histo ){
    histoman.storeEventSetup( &e, &eSetup );
  }

  phase1analyze( e, eSetup );
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
    histoman.initTriggerFlag( run, iSetup );// runnig trigger flag initialization
  }
}
