#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"

// -----------------------------------------------------------------------------
//
PedestalsTask::PedestalsTask( DaqMonitorBEInterface* dqm,
			      const SiStripModule& module ) :
  CommissioningTask( dqm, module ),
  peds_()
{
  cout << "[PedestalsTask::PedestalsTask]" 
       << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
PedestalsTask::~PedestalsTask() {
  cout << "[PedestalsTask::~PedestalsTask]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::book( const SiStripModule& module ) {
  cout << "[PedestalsTask::book]" << endl;
  
  unsigned short nbins = 256 * module.nPairs();

  peds_.meSumOfSquares_  = dqm_->book1D( "RawPedestals|sumOfSquares",  "RawPedestals|sumOfSquares",  nbins, 0., nbins*1. );
  peds_.meSumOfContents_ = dqm_->book1D( "RawPedestals|sumOfContents", "RawPedestals|sumOfContents", nbins, 0., nbins*1. );
  peds_.meNumOfEntries_  = dqm_->book1D( "RawPedestals|numOfEntries",  "RawPedestals|numOfEntries",  nbins, 0., nbins*1. );
  
  peds_.vSumOfSquares_.resize(nbins,0);
  peds_.vSumOfContents_.resize(nbins,0);
  peds_.vNumOfEntries_.resize(nbins,0);
  
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  cout << "[PedestalsTask::fill]" << endl;

  if ( digis.data.size() != peds_.vNumOfEntries_.size() ) {
    cerr << "[PedestalsTask::fill]" 
	 << " Unexpected number of digis! " 
	 << peds_.vNumOfEntries_.size() << endl; 
  }

  // Check number of digis
  unsigned short nbins = peds_.vNumOfEntries_.size();
  if ( digis.data.size() < nbins ) { nbins = digis.data.size(); }
  
  // Fill vectors
  for ( unsigned short ibin = 0; ibin < nbins; ibin++ ) {
    peds_.vSumOfSquares_[ibin] += digis.data[ibin].adc() * digis.data[ibin].adc();
    peds_.vSumOfContents_[ibin] += digis.data[ibin].adc();
    peds_.vNumOfEntries_[ibin]++;
  }      

}

// -----------------------------------------------------------------------------
//
void PedestalsTask::update() {
  cout << "[PedestalsTask::update]" << endl;
  for ( unsigned short ibin = 0; ibin < peds_.vNumOfEntries_.size(); ibin++ ) {
    peds_.meSumOfSquares_->setBinContent( ibin+1, peds_.vSumOfSquares_[ibin] );
    peds_.meSumOfContents_->setBinContent( ibin+1, peds_.vSumOfContents_[ibin] );
    peds_.meNumOfEntries_->setBinContent( ibin+1, peds_.vNumOfEntries_[ibin] );
  }
}


