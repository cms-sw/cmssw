#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"

// -----------------------------------------------------------------------------
//
ApvTimingTask::ApvTimingTask( DaqMonitorBEInterface* dqm,
			      const SiStripModule& module ) :
  CommissioningTask( dqm, module ),
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
void ApvTimingTask::book( const SiStripModule& module ) {
  cout << "[ApvTimingTask::book]" << endl;
  
  unsigned short nbins = 60; //@@ correct?

  timing_.resize( module.nPairs(), HistoSet() );
  for ( unsigned short ipair = 0; ipair < module.nPairs(); ipair++ ) {
    //@@ need to reorder pairs here? 
    stringstream ss;

    ss.str(""); ss << "ApvTiming|ApvPair" << ipair << "|sumOfSquares";
    timing_[ipair].meSumOfSquares_  = dqm_->book1D( ss.str(), ss.str(), nbins, 0., nbins*1. );
    ss.str(""); ss << "ApvTiming|ApvPair" << ipair << "|sumOfContents";
    timing_[ipair].meSumOfContents_ = dqm_->book1D( ss.str(), ss.str(), nbins, 0., nbins*1. );
    ss.str(""); ss << "ApvTiming|ApvPair" << ipair << "|numOfEntries";
    timing_[ipair].meNumOfEntries_  = dqm_->book1D( ss.str(), ss.str(), nbins, 0., nbins*1. );

    timing_[ipair].vSumOfSquares_.resize(nbins,0);
    timing_[ipair].vSumOfContents_.resize(nbins,0);
    timing_[ipair].vNumOfEntries_.resize(nbins,0);
  }
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  cout << "[ApvTimingTask::fill]" << endl;

  //@@ get bin number from SiStripEventInfo!
  unsigned short ibin = 0;

  if ( digis.data.size() != 256*timing_.size() ) {
    cerr << "[ApvTimingTask::fill]" 
	 << " Number of digis (" << digis.data.size() 
	 << ") is not compatible with number of APV pairs ("
	 << timing_.size() << ")!" << endl; 
  }

  // Fill vectors
  for ( unsigned short idigi = 0; idigi < digis.data.size(); idigi++ ) {
    timing_[idigi/256].vSumOfSquares_[ibin] += digis.data[idigi].adc() * digis.data[idigi].adc();
    timing_[idigi/256].vSumOfContents_[ibin] += digis.data[idigi].adc();
    timing_[idigi/256].vNumOfEntries_[ibin]++;
  }      
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::update() {
  cout << "[ApvTimingTask::update]" << endl;

  for ( unsigned short ipair = 0; ipair < timing_.size(); ipair++ ) {
    for ( unsigned short ibin = 0; ibin < timing_[ipair].vNumOfEntries_.size(); ibin++ ) {
      timing_[ipair].meSumOfSquares_->setBinContent(  ibin+1, timing_[ipair].vSumOfSquares_[ibin]*1. );
      timing_[ipair].meSumOfContents_->setBinContent( ibin+1, timing_[ipair].vSumOfContents_[ibin]*1. );
      timing_[ipair].meNumOfEntries_->setBinContent(  ibin+1, timing_[ipair].vNumOfEntries_[ibin]*1. );
    }
  }

}


