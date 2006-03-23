#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"

// -----------------------------------------------------------------------------
//
ApvTimingTask::ApvTimingTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn ),
  timing_()
{
  cout << "[ApvTimingTask::ApvTimingTask]" 
       << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
ApvTimingTask::~ApvTimingTask() {
  cout << "[ApvTimingTask::~ApvTimingTask]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::book( const FedChannelConnection& conn ) {
  cout << "[ApvTimingTask::book]" << endl;
  
  unsigned short nbins = 60; //@@ correct?

  stringstream ss;
  ss.str(""); ss << "ApvTiming|ApvPair" << conn.lldChannel() << "|sumOfSquares";
  timing_.meSumOfSquares_  = dqm_->book1D( ss.str(), ss.str(), nbins, 0., nbins*1. );
  ss.str(""); ss << "ApvTiming|ApvPair" << conn.lldChannel() << "|sumOfContents";
  timing_.meSumOfContents_ = dqm_->book1D( ss.str(), ss.str(), nbins, 0., nbins*1. );
  ss.str(""); ss << "ApvTiming|ApvPair" << conn.lldChannel() << "|numOfEntries";
  timing_.meNumOfEntries_  = dqm_->book1D( ss.str(), ss.str(), nbins, 0., nbins*1. );
  
  timing_.vSumOfSquares_.resize(nbins,0);
  timing_.vSumOfContents_.resize(nbins,0);
  timing_.vNumOfEntries_.resize(nbins,0);
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  cout << "[ApvTimingTask::fill]" << endl;

  //@@ get bin number from SiStripEventInfo!
  unsigned short ibin = 0;

  if ( digis.data.size() != 256 ) {
    cerr << "[ApvTimingTask::fill]" 
	 << " Unexpected number of digis! " << digis.data.size() 
	 << endl; 
  }
  
  // Fill vectors
  for ( unsigned short idigi = 0; idigi < digis.data.size(); idigi++ ) {
    timing_.vSumOfSquares_[ibin] += digis.data[idigi].adc() * digis.data[idigi].adc();
    timing_.vSumOfContents_[ibin] += digis.data[idigi].adc();
    timing_.vNumOfEntries_[ibin]++;
  }      
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::update() {
  cout << "[ApvTimingTask::update]" << endl;

  for ( unsigned short ibin = 0; ibin < timing_.vNumOfEntries_.size(); ibin++ ) {
    timing_.meSumOfSquares_->setBinContent(  ibin+1, timing_.vSumOfSquares_[ibin]*1. );
    timing_.meSumOfContents_->setBinContent( ibin+1, timing_.vSumOfContents_[ibin]*1. );
    timing_.meNumOfEntries_->setBinContent(  ibin+1, timing_.vNumOfEntries_[ibin]*1. );
  }
  
}


