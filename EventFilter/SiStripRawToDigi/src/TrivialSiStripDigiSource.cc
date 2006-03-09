#include "EventFilter/SiStripRawToDigi/interface/TrivialSiStripDigiSource.h"
// edm 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
// cabling
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// fed
#include "Fed9UUtils.hh"
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

using namespace std;

// -----------------------------------------------------------------------------
/** */
TrivialSiStripDigiSource::TrivialSiStripDigiSource( const edm::ParameterSet& pset ) :
  eventCounter_(0),
  meanOcc_( pset.getUntrackedParameter<double>("MeanOccupancy",1.0) ),
  rmsOcc_( pset.getUntrackedParameter<double>("RmsOccupancy",0.1) )
{
  cout << "[TrivialSiStripDigiSource::TrivialSiStripDigiSource]"
       << " Constructing object..." << endl;

  //srand( time( NULL ) ); // seed for random number generator
  produces< edm::DetSetVector<SiStripDigi> >();
}

// -----------------------------------------------------------------------------
/** */
TrivialSiStripDigiSource::~TrivialSiStripDigiSource() {
  cout << "[TrivialSiStripDigiSource::~TrivialSiStripDigiSource]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void TrivialSiStripDigiSource::produce( edm::Event& iEvent, 
					const edm::EventSetup& iSetup ) {
  
  eventCounter_++; 
  cout << "[TrivialSiStripDigiSource::produce] "
       << "event number: " << eventCounter_ << endl;
  
  edm::ESHandle<SiStripFedCabling> cabling;
  iSetup.get<SiStripFedCablingRcd>().get( cabling );

  auto_ptr< edm::DetSetVector<SiStripDigi> > collection( new edm::DetSetVector<SiStripDigi> );
  
  uint32_t nchans = 0;
  uint32_t ndigis = 0;
  const vector<uint16_t>& fed_ids = cabling->feds(); 
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {
    for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
      const FedChannelConnection& conn = cabling->connection( *ifed, ichan );
      if ( conn.detId() ) {
	nchans++;
	edm::DetSet<SiStripDigi>& digis = collection->find_or_insert( conn.detId() );
	uint16_t ndigi = static_cast<uint16_t>( 2.56 * RandGauss::shoot( meanOcc_, rmsOcc_ ) );
	vector<int> used_strips; used_strips.reserve(ndigi);
	vector<int>::iterator iter;
	unsigned int idigi = 0;
	while ( idigi < ndigi ) {
	  int adc = static_cast<uint16_t>( 1024. * RandFlat::shoot() ) + 1;
	  int str = static_cast<uint16_t>( 256. * RandFlat::shoot() );
	  iter = find( used_strips.begin(), used_strips.end(), str );
	  if ( iter == used_strips.end() ) { 
	    digis.data.push_back( SiStripDigi( str, adc ) );
	    used_strips.push_back( str ); 
	    ndigis++;
	    idigi++;
	  }
	}
      }
    }
  }
  
  iEvent.put( collection );
  
  if ( nchans ) { 
    cout << "[TrivialSiStripDigiSource::produce]"
	 << " Generated " << ndigis 
	 << " digis for " << nchans
	 << " channels with a mean occupancy of " 
	 << dec << setprecision(2)
	 << ( 1. / 2.56 ) * (float)ndigis / (float)nchans << " %"
	 << endl;
  }
  
}
