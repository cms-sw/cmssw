#include "DQM/SiStripCommissioningSources/interface/PedsOnlyTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/UpdateTProfile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <math.h>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
PedsOnlyTask::PedsOnlyTask( DQMStore* dqm,
			    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "PedsOnlyTask" ),
  peds_()
{
  LogTrace(mlDqmSource_)
    << "[PedsOnlyTask::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
PedsOnlyTask::~PedsOnlyTask() {
  LogTrace(mlDqmSource_)
    << "[PedsOnlyTask::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void PedsOnlyTask::book() {
  LogTrace(mlDqmSource_) << "[PedsOnlyTask::" << __func__ << "]";
  
  uint16_t nbins;
  std::string title;
  std::string extra_info;
  peds_.resize(2);
  nbins = 256;
  
  // Pedestals histogram
  extra_info = sistrip::extrainfo::pedestals_; 
  peds_[0].isProfile_ = true;
  
  title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			     sistrip::PEDS_ONLY, 
			     sistrip::FED_KEY, 
			     fedKey(),
			     sistrip::LLD_CHAN, 
			     connection().lldChannel(),
			     extra_info ).title();
  
  peds_[0].histo_ = dqm()->bookProfile( title, title, 
					nbins, -0.5, nbins*1.-0.5,
					1025, 0., 1025. );
  
  peds_[0].vNumOfEntries_.resize(nbins,0);
  peds_[0].vSumOfContents_.resize(nbins,0);
  peds_[0].vSumOfSquares_.resize(nbins,0);

  // Raw noise histogram
  extra_info = sistrip::extrainfo::rawNoise_; 
  peds_[1].isProfile_ = true;
  
  title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			     sistrip::PEDS_ONLY, 
			     sistrip::FED_KEY, 
			     fedKey(),
			     sistrip::LLD_CHAN, 
			     connection().lldChannel(),
			     extra_info ).title();
  
  peds_[1].histo_ = dqm()->bookProfile( title, title, 
   					nbins, -0.5, nbins*1.-0.5,
   					1025, 0., 1025. );
  
  peds_[1].vNumOfEntries_.resize(nbins,0);
  peds_[1].vSumOfContents_.resize(nbins,0);
  peds_[1].vSumOfSquares_.resize(nbins,0);
  
}

// -----------------------------------------------------------------------------
//
void PedsOnlyTask::fill( const SiStripEventSummary& summary,
			 const edm::DetSet<SiStripRawDigi>& digis ) {
  
  if ( digis.data.size() != peds_[0].vNumOfEntries_.size() ) {
    edm::LogWarning(mlDqmSource_)
      << "[PedsOnlyTask::" << __func__ << "]"
      << " Unexpected number of digis: " 
      << digis.data.size(); 
    return;
  }
  
  // Check number of digis
  uint16_t nbins = peds_[0].vNumOfEntries_.size();
  if ( digis.data.size() < nbins ) { nbins = digis.data.size(); }

  //@@ PEDS ALGORITHM HERE!!!
  //@@ PEDS ALGORITHM HERE!!!
  //@@ PEDS ALGORITHM HERE!!!
  
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    // pedestals
    updateHistoSet( peds_[0], ibin, digis.data[ibin].adc() ); 
    // raw noise (QUICK FIX HERE!)
    updateHistoSet( peds_[1], ibin, 0. ); //@@ use any value for now as filled in update() method below!
  }
  
}

// -----------------------------------------------------------------------------
//
void PedsOnlyTask::update() {
  
  // Pedestals histogram
  updateHistoSet( peds_[0] );
  
  // Raw noise histogram (QUICK FIX HERE!)
  TProfile* histo = ExtractTObject<TProfile>().extract( peds_[1].histo_ );
  for ( uint16_t ii = 0; ii < peds_[0].vNumOfEntries_.size(); ++ii ) {
    float entries =  peds_[0].vNumOfEntries_[ii];
    if ( entries > 0. ) {
      float mean   = peds_[0].vSumOfContents_[ii] / peds_[0].vNumOfEntries_[ii];
      float spread = sqrt( fabs( peds_[0].vSumOfSquares_[ii] / peds_[0].vNumOfEntries_[ii] - mean * mean ) );
      float error  = 0; // sqrt(entries) / entries;
      UpdateTProfile::setBinContent( histo, ii+1, entries, spread, error );
    }
  }
  
}

