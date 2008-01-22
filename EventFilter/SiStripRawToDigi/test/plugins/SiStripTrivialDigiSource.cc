// Last commit: $Id: SiStripTrivialDigiSource.cc,v 1.4 2007/04/30 13:54:19 pwing Exp $

#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialDigiSource.h"
// edm 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
// cabling
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// fed
// clhep
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"
// std
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripTrivialDigiSource::SiStripTrivialDigiSource( const edm::ParameterSet& pset ) :
  eventCounter_(0),
  testDistr_( pset.getUntrackedParameter<bool>("TestDistribution",false) ),
  meanOcc_( pset.getUntrackedParameter<double>("MeanOccupancy",1.0) ),
  rmsOcc_( pset.getUntrackedParameter<double>("RmsOccupancy",0.1) ),
  anal_()
{
  LogDebug("TrivialDigiSource") << "[SiStripTrivialDigiSource::SiStripTrivialDigiSource] Constructing object...";
  
  //srand( time( NULL ) ); // seed for random number generator
  produces< edm::DetSetVector<SiStripDigi> >();
}

// -----------------------------------------------------------------------------
/** */
SiStripTrivialDigiSource::~SiStripTrivialDigiSource() {
  LogDebug("TrivialDigiSource") << "[SiStripTrivialDigiSource::~SiStripTrivialDigiSource] Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialDigiSource::produce( edm::Event& iEvent, 
					const edm::EventSetup& iSetup ) {
  
  eventCounter_++; 
  LogDebug("TrivialDigiSource") << "[SiStripTrivialDigiSource::produce] Event: " << eventCounter_;
  //anal_.addEvent();
  
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );
  
  auto_ptr< edm::DetSetVector<SiStripDigi> > collection( new edm::DetSetVector<SiStripDigi> );
  
  uint32_t nchans = 0;
  uint32_t ndigis = 0;
  const vector<uint16_t>& fed_ids = cabling->feds(); 
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
    //anal_.addFed();
    for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
      const FedChannelConnection& conn = cabling->connection( *ifed, ichan );
      // Check DetID is non-zero and valid
      if (!conn.detId() ||
	  (conn.detId() == sistrip::invalid32_)) { continue; }
      //anal_.addChan(); nchans++;
      edm::DetSet<SiStripDigi>& digis = collection->find_or_insert( conn.detId() );
      uint16_t ngroups = 1;
      uint16_t ndigi;
      if ( testDistr_ ) { ndigi = 256 / ngroups; }
      else {
	float rdm = 2.56 * RandGauss::shoot( meanOcc_, rmsOcc_ );
	float tmp; bool extra = ( RandFlat::shoot() > modf(rdm,&tmp) );
	ndigi = static_cast<uint16_t>(rdm) + static_cast<uint16_t>(extra);
      }
      vector<uint16_t> used_strips; used_strips.reserve(ndigi);
      vector<uint16_t>::iterator iter;
      uint16_t idigi = 0;
      while ( idigi < ndigi ) {
	uint16_t str;
	uint16_t adc;
	if ( testDistr_ ) { str = idigi*ngroups; adc = (idigi+1)*ngroups-1; }
	else {
	  str = static_cast<uint16_t>( 256. * RandFlat::shoot() );
	  adc = static_cast<uint16_t>( 256. * RandFlat::shoot() );
	}
	iter = find( used_strips.begin(), used_strips.end(), str );
	if ( iter == used_strips.end() && adc ) { // require non-zero adc!
	  digis.data.push_back( SiStripDigi( str+conn.apvPairNumber()*256, adc ) );
	  used_strips.push_back( str ); 
	  //anal_.zsDigi( str+conn.apvPairNumber()*256, adc );
	  ndigis++;
	  idigi++;
	}
      }
    }
  }
  
  iEvent.put( collection );
  
  if ( nchans ) { 
    stringstream ss;
    ss << "[SiStripTrivialDigiSource::produce]"
       << " Generated " << ndigis 
       << " digis for " << nchans
       << " channels with a mean occupancy of " 
       << dec << setprecision(2)
       << ( 1. / 2.56 ) * (float)ndigis / (float)nchans << " %";
    LogDebug("TrivialDigiSource") << ss.str();
  }
  
}
