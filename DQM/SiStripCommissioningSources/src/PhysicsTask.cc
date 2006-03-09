#include "DQM/SiStripCommissioningSources/interface/PhysicsTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"

// -----------------------------------------------------------------------------
//
PhysicsTask::PhysicsTask( DaqMonitorBEInterface* dqm,
			  const SiStripModule& module ) :
  CommissioningTask( dqm, module ),
  landau_()
{
  cout << "[PhysicsTask::PhysicsTask]" 
       << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
PhysicsTask::~PhysicsTask() {
  cout << "[PhysicsTask::~PhysicsTask]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
void PhysicsTask::book( const SiStripModule& module ) {
  cout << "[PhysicsTask::book]" << endl;
  
  unsigned short nbins = 1024;

  landau_.meSumOfSquares_  = dqm_->book1D( "Landau|sumOfSquares",  "Landau|sumOfSquares",  nbins, 0., nbins*1. );
  landau_.meSumOfContents_ = dqm_->book1D( "Landau|sumOfContents", "Landau|sumOfContents", nbins, 0., nbins*1. );
  landau_.meNumOfEntries_  = dqm_->book1D( "Landau|numOfEntries",  "Landau|numOfEntries",  nbins, 0., nbins*1. );

  landau_.vSumOfSquares_.resize(nbins,0);
  landau_.vSumOfContents_.resize(nbins,0);
  landau_.vNumOfEntries_.resize(nbins,0);
  
}

// -----------------------------------------------------------------------------
//
void PhysicsTask::fill( const SiStripEventSummary& summary,
			const edm::DetSet<SiStripRawDigi>& digis ) {
  cout << "[PhysicsTask::fill]" << endl;

  // Fill vectors
  for ( unsigned short ibin = 0; ibin < landau_.vNumOfEntries_.size(); ibin++ ) {
    landau_.vSumOfSquares_[ibin] += digis.data[ibin].adc() * digis.data[ibin].adc();
    landau_.vSumOfContents_[ibin] += digis.data[ibin].adc();
    landau_.vNumOfEntries_[ibin]++;
  }

}

// -----------------------------------------------------------------------------
//
void PhysicsTask::update() {
  cout << "[PhysicsTask::update]" << endl;
  for ( unsigned short ibin = 0; ibin < landau_.vNumOfEntries_.size(); ibin++ ) {
    landau_.meSumOfSquares_->setBinContent( ibin+1, landau_.vSumOfSquares_[ibin]*1. );
    landau_.meSumOfContents_->setBinContent( ibin+1, landau_.vSumOfContents_[ibin]*1. );
    landau_.meNumOfEntries_->setBinContent( ibin+1, landau_.vNumOfEntries_[ibin]*1. );
  }
}
