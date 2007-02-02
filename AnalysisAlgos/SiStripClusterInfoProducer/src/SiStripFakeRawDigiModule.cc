#include "AnalysisAlgos/SiStripClusterInfoProducer/interface/SiStripFakeRawDigiModule.h"
// 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// 
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
//
#include "boost/cstdint.hpp"
#include <cstdlib>

// -----------------------------------------------------------------------------
//
SiStripFakeRawDigiModule::SiStripFakeRawDigiModule( const edm::ParameterSet& pset ) {

  produces< SiStripEventSummary >();
  produces< edm::DetSetVector<SiStripRawDigi> >("ScopeMode");
  produces< edm::DetSetVector<SiStripRawDigi> >("VirginRaw");
  produces< edm::DetSetVector<SiStripRawDigi> >("ProcessedRaw");
}

void SiStripFakeRawDigiModule::produce( edm::Event& event, 
				      const edm::EventSetup& setup ) {
  
  std::auto_ptr<SiStripEventSummary> summary( new SiStripEventSummary() );
  
  // Create std::auto pointers for digi products
  std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > sm( new edm::DetSetVector<SiStripRawDigi> );
  std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > vr( new edm::DetSetVector<SiStripRawDigi> );
  std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > pr( new edm::DetSetVector<SiStripRawDigi> );

  event.put( summary );
  event.put( sm, "ScopeMode" );
  event.put( vr, "VirginRaw" );
  event.put( pr, "ProcessedRaw" );
}
